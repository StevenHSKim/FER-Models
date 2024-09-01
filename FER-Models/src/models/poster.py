'''
POSTER 모델 정의
'''

import torch
import collections
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from pretrain.hyp_crossvit import HyVisionTransformer
from pretrain.mobilefacenet import MobileFaceNet
from pretrain.ir50 import Backbone
from common.loss import LabelSmoothingCrossEntropy


class POSTER(nn.Module):
    def __init__(self, config):
        super(POSTER, self).__init__()

        # 설정에서 필요한 매개변수 추출
        img_size = config['img_size']
        num_classes = config['num_classes']
        model_type = config['model_type']
        depth = config['depth'][model_type]

        self.img_size = img_size
        self.num_classes = num_classes

        # MobileFaceNet 초기화 및 사전 학습된 가중치 로드
        self.face_landback = MobileFaceNet([112, 112], 136)
        face_landback_checkpoint = torch.load(config['face_landback_checkpoint'], map_location=lambda storage, loc: storage)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])
        for param in self.face_landback.parameters():
            param.requires_grad = False

        # Backbone IR 모델 초기화 및 가중치 로드
        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load(config['ir_checkpoint'], map_location=lambda storage, loc: storage)
        self.ir_back = self.load_pretrained_weights(self.ir_back, ir_checkpoint)

        # 추가 모델 구성요소 초기화
        self.ir_layer = nn.Linear(1024, 512)
        self.pyramid_fuse = HyVisionTransformer(
            in_chans=49, q_chanel=49, embed_dim=512,
            depth=depth, num_heads=8, mlp_ratio=2.,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1
        )
        self.se_block = SE_block(input_dim=512)
        self.head = ClassificationHead(input_dim=512, target_dim=num_classes)

        # 손실 함수 초기화
        self.ce_loss = nn.CrossEntropyLoss()
        self.lsce_loss = LabelSmoothingCrossEntropy(smoothing=0.1)

    def forward(self, x):
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1, 2)
        x_ir = self.ir_back(x)
        x_ir = self.ir_layer(x_ir)
        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)
        return out, y_feat

    def calculate_loss(self, outputs, targets):
        CE_loss = self.ce_loss(outputs, targets)
        lsce_loss = self.lsce_loss(outputs, targets)
        loss = 2 * lsce_loss + CE_loss
        return loss

    def load_pretrained_weights(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint.get('state_dict', checkpoint)
        model_state_dict = model.state_dict()
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if k in model_state_dict and model_state_dict[k].size() == v.size():
                new_state_dict[k] = v
            else:
                print(f"Skipping layer {k} due to size mismatch or it doesn't exist in the model.")
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        return model

class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super(SE_block, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmoid(x1)
        x = x * x1
        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super(ClassificationHead, self).__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat
