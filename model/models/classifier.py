import torch
import torch.nn as nn
import numpy as np
from model.utils import euclidean_metric
import torch.nn.functional as F
    
class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            hdim = 64
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)                        
        else:
            raise ValueError('')

        # self.fc = nn.Linear(hdim, args.num_class)

        self.cls_classifier = nn.Linear(hdim, args.num_class)
        self.rot_classifier = nn.Linear(args.num_class, 4)

    def forward(self, generated_image, is_emb = False):
        # print("generated_image: ", generated_image.size())    # [128, 3, 84, 84]

        feat = self.encoder(generated_image)
        output = self.cls_classifier(feat)
        rot_output = self.rot_classifier(output)

        return output, rot_output

        # if not is_emb:
            # out = self.fc(out)
        # return out
    
    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        query = self.encoder(data_query)
        
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim
    