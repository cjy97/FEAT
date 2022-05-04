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
    ):
        super(DistillLayer, self).__init__()
        self.encoder = self._load_state_dict(teacher_backbone_class, teacher_init_weights, is_distill)

    def _load_state_dict(self, teacher_backbone_class, teacher_init_weights, is_distill):
        new_model = None

        if is_distill and teacher_init_weights is not None:
            if teacher_backbone_class == 'Res12':
                new_model = ResNet()
            else:
                raise ValueError('')
            
            model_dict = new_model.state_dict()

            pretrained_dict = torch.load(teacher_init_weights)['params']
            pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items() if k.replace("encoder.", "") in model_dict}
            
            for k,_ in pretrained_dict.items():
                print("pretrained key: ", k)

            model_dict.update(pretrained_dict)
            new_model.load_state_dict(model_dict)   # 只将权重加载给教师模型

        return new_model
        

    @torch.no_grad()
    def forward(self, x):
        local_features = None
        if self.encoder is not None:
            local_features = self.encoder(x)
        
        return local_features