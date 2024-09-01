'''
DAN 모델 정의
'''

from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models

from common.loss import AffinityLoss, PartitionLoss  # 손실 함수 임포트

class DAN(nn.Module):
    def __init__(self, config):
        super(DAN, self).__init__()
        
        resnet = models.resnet18(pretrained=config['pretrained'])
        
        if config['pretrained']:
            checkpoint = torch.load(config['resnet18_checkpoint_path'])
            resnet.load_state_dict(checkpoint['state_dict'], strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = config['num_head']
        
        for i in range(self.num_head):
            setattr(self, "cat_head%d" % i, CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, config['num_class'])
        self.bn = nn.BatchNorm1d(config['num_class'])
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.af_loss = AffinityLoss(device=config['device'], num_class=config['num_class'], feat_dim=512)
        self.pt_loss = PartitionLoss()

    def forward(self, x):
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))
        
        heads = torch.stack(heads).permute([1, 0, 2])
        if heads.size(1) > 1:
            heads = F.log_softmax(heads, dim=1)
            
        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)
   
        return out, x, heads

    def calculate_loss(self, outputs, targets):
        # outputs는 forward 함수에서 반환된 out, x, heads를 포함합니다.
        out, feat, heads = outputs

        # 각 손실 계산
        ce_loss = self.ce_loss(out, targets)
        af_loss = self.af_loss(feat, targets)
        pt_loss = self.pt_loss(heads)

        # 총 손실 계산
        total_loss = ce_loss + af_loss + pt_loss

        return total_loss

    def load_pretrained_weights(self, checkpoint_path):
        """
        DataParallel을 사용하지 않고 사전 학습된 가중치를 로드하는 함수
        """
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 현재 모델의 state_dict 가져오기
        model_state_dict = self.state_dict()
        
        # 새로운 state_dict에 가중치 매핑
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and model_state_dict[k].size() == v.size():
                new_state_dict[k] = v
            else:
                print(f"Skipping {k} as size mismatch or not found in model.")
        
        # 모델에 가중치 로드
        model_state_dict.update(new_state_dict)
        self.load_state_dict(model_state_dict)
        print("Pretrained weights loaded successfully.")
        
        
class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)
        return ca

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y
        return out

class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()    
        )

    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0), -1)
        y = self.attention(sa)
        out = sa * y
        return out
