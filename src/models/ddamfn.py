'''
DDAMFN 모델 정의
'''

import os
import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
import collections

from pretrain import MixedFeatureNet
from common.loss import AttentionLoss  # 손실 함수 임포트


class DDAMNet(nn.Module):
    def __init__(self, config):
        super(DDAMNet, self).__init__()

        net = MixedFeatureNet.MixedFeatureNet()

        if config['pretrained']:
            # Assuming checkpoint includes only state_dict under 'state_dict' key
            checkpoint = torch.load(config['pretrained_model_path'])
            net.load_state_dict(checkpoint['state_dict'], strict=True)

        self.features = nn.Sequential(*list(net.children())[:-4])
        self.num_head = config['num_head']
        for i in range(self.num_head):
            setattr(self, "cat_head%d" % i, CoordAttHead())

        self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()
        self.fc = nn.Linear(512, config['num_class'])
        self.bn = nn.BatchNorm1d(config['num_class'])

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.at_loss = AttentionLoss(device=config['device'])

    def forward(self, x):
        x = self.features(x)
        heads = []
       
        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))
        head_out = heads
        
        y = heads[0]
        
        for i in range(1, self.num_head):
            y = torch.max(y, heads[i])                     
        
        y = x * y
        y = self.Linear(y)
        y = self.flatten(y) 
        out = self.fc(y)        
        return out, x, head_out

    def calculate_loss(self, outputs, targets):
        out, feat, heads = outputs

        # 각 손실 계산
        ce_loss = self.ce_loss(out, targets)
        at_loss = 0.1 * self.at_loss(heads)

        # 총 손실 계산
        total_loss = ce_loss + at_loss

        return total_loss
    

    def load_pretrained_weights(model, checkpoint_path):
        """
        사전 학습된 가중치를 모델에 로드하는 함수.
        DataParallel로 저장된 가중치에서 'module.' 접두사를 제거한 후 로드.
        """
        # Checkpoint 로드
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 현재 모델의 state_dict 가져오기
        model_state_dict = model.state_dict()
        
        # 새로운 state_dict에 가중치 매핑
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            # 'module.' 접두사 제거
            if k.startswith('module.'):
                k = k[7:]
            # 모델의 state_dict와 일치하는 경우에만 업데이트
            if k in model_state_dict and model_state_dict[k].size() == v.size():
                new_state_dict[k] = v
            else:
                print(f"Skipping layer {k} due to size mismatch or it doesn't exist in the model.")
        
        # 모델에 가중치 로드
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        
        print("Pretrained weights loaded successfully.")
        return model

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
                      
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordAtt(512, 512)
    def forward(self, x):
        ca = self.CoordAtt(x)
        return ca  
        
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
      
        self.Linear_h = Linear_block(inp, inp, groups=inp, kernel=(1, 7), stride=(1, 1), padding=(0, 0))        
        self.Linear_w = Linear_block(inp, inp, groups=inp, kernel=(7, 1), stride=(1, 1), padding=(0, 0))
        
        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(oup, oup, groups=oup, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten() 

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.Linear_h(x)
        x_w = self.Linear_w(x)
        x_w = x_w.permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        
        y = x_w * x_h
 
        return y
