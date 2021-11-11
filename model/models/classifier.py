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

        self.fc = nn.Linear(hdim, args.num_class)
        
        self.dn4_layer = DN4Layer(16, self.args.shot, self.args.query, n_k = 5)
        self.avgpool = nn.AvgPool2d(5, stride=1)

    def forward(self, data, is_emb = False):
        out = self.encoder(data)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        if not is_emb:
            out = self.fc(out)

        return out
    
    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class
        support = self.encoder(data_shot)
        query = self.encoder(data_query)
        
        # print("support: ", support.size())  # [16, 640, 5, 5]
        # print("query: ", query.size())      # [240, 640, 5, 5]
        _, emb_dim, h, w = support.size()
        
        support_ = support.view(1, self.args.shot, way, emb_dim, h, w)
        support_ = support_.permute(0, 2, 1, 3, 4, 5)
        support_ = support_.contiguous().view(1, way*self.args.shot, emb_dim, h, w)
        query_ = query.view(1, self.args.query*way, emb_dim, h, w)
        logits_dn4 = self.dn4_layer(query_, support_).view(1*way*self.args.query, way)
        
        
        proto = self.avgpool(support)
        proto = proto.view(proto.size(0), -1)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        query = self.avgpool(query)
        query = query.view(query.size(0), -1)
        
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        
        return logits_dist, logits_sim, logits_dn4


class DN4Layer(nn.Module):
    def __init__(self, way_num, shot_num, query_num, n_k):
        super(DN4Layer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.n_k = n_k

    def forward(self, query_feat, support_feat):
        t, wq, c, h, w = query_feat.size()
        _, ws, _, _, _ = support_feat.size()

        # t, wq, c, hw -> t, wq, hw, c -> t, wq, 1, hw, c
        query_feat = query_feat.view(t, self.way_num * self.query_num, c, h * w) \
            .permute(0, 1, 3, 2)
        query_feat = F.normalize(query_feat, p=2, dim=2).unsqueeze(2)

        # t, ws, c, h, w -> t, w, s, c, hw -> t, 1, w, c, shw
        support_feat = support_feat.view(t, self.way_num, self.shot_num, c, h * w) \
            .permute(0, 1, 3, 2, 4).contiguous() \
            .view(t, self.way_num, c, self.shot_num * h * w)
        support_feat = F.normalize(support_feat, p=2, dim=2).unsqueeze(1)

        # t, wq, w, hw, shw -> t, wq, w, hw, n_k -> t, wq, w
        relation = torch.matmul(query_feat, support_feat)
        # print("relation: ", relation.size())        # [1, 75, 5, 25, 25]
        topk_value, _ = torch.topk(relation, self.n_k, dim=-1)
        # print("topk_value: ", topk_value.size())    # [1, 75, 5, 25, 3]
        score = torch.sum(topk_value, dim=[3, 4])
        # print("dn4 output score: ", score.size())   # [1, 75, 5]
        return score