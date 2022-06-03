import copy
import torch
import torch.nn as nn

from model.networks.res12 import ResNet

class DistillLayer(nn.Module):
    def __init__(
        self,
        teacher_backbone_class,
        teacher_init_weights,
        is_distill,
        kd_loss,
    ):
        super(DistillLayer, self).__init__()
        self.encoder = self._load_state_dict(teacher_backbone_class, teacher_init_weights, is_distill, type = "encoder.")

        if kd_loss == "KD":     # 只有基于分类logits的传统蒸馏方法（KD），教师模型需要加载线性分类头
            self.fc = self._load_state_dict(teacher_backbone_class, teacher_init_weights, is_distill, type = "fc.")
            self.GAP = nn.AvgPool2d(5, stride=1)
        else:
            self.fc = None

    def _load_state_dict(self, teacher_backbone_class, teacher_init_weights, is_distill, type):
        new_model = None

        if is_distill and teacher_init_weights is not None:
            
            if type == "encoder.":
                if teacher_backbone_class == 'Res12':
                    new_model = ResNet()
            elif type == "fc.":
                if teacher_backbone_class == 'Res12':
                    new_model = nn.Linear(640, 64)
            
            model_dict = new_model.state_dict()

            pretrained_dict = torch.load(teacher_init_weights)['params']
            pretrained_dict = {k.replace(type, ""): v for k, v in pretrained_dict.items() if k.replace(type, "") in model_dict}

            model_dict.update(pretrained_dict)
            new_model.load_state_dict(model_dict)   # 只将权重加载给教师模型

            for k, _ in pretrained_dict.items():
                print("pretrained key: ", k)
            for key, _ in pretrained_dict.items():
                print("key: ", key)

        return new_model
        

    @torch.no_grad()
    def forward(self, x):
        local_features = None

        if self.encoder is not None:
            local_features = self.encoder(x)
        
        if self.fc is not None:
            features = self.GAP(local_features)
            logits = self.fc( features.view(features.size(0), -1) )
            return logits
        else:
            return local_features
