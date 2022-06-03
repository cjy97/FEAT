import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel
from model.models.DistillLayer import DistillLayer
from model.models.Prune import prune_resnet


class DN4(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

        # if args.backbone_class == 'ConvNet':
        #     hdim = 64
        # elif args.backbone_class == 'Res12':
        #     hdim = 640
        # elif args.backbone_class == 'Res18':
        #     hdim = 512
        # elif args.backbone_class == 'WRN':
        #     hdim = 640
        # else:
        #     raise ValueError('')

        """ # 加载学生模型预训练权重（如果有的话）
        if args.init_weights is not None:
            model_dict = self.encoder.state_dict()        
            pretrained_dict = torch.load(args.init_weights)['params']
            if args.backbone_class == 'ConvNet':
                pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items() if k.replace("encoder.", "") in model_dict}
            print("load init_weights: ", pretrained_dict.keys())
            model_dict.update(pretrained_dict)
            self.encoder.load_state_dict(model_dict)
        """
        
        self.dn4_layer = DN4Layer(args.way, args.shot, args.query, n_k = 3)

        self.GAP = nn.AvgPool2d(5, stride=1)
        self.fc = nn.Linear(640, 64)

        self.distill_layer = DistillLayer(
            args.teacher_backbone_class,
            args.teacher_init_weights,
            args.is_distill,
            args.kd_loss,
        ).requires_grad_(False)

        if args.is_prune:   # 如果指定了“剪枝”模式，则重置学生模型
            self.encoder = prune_resnet(self.encoder, args.remove_ratio)
    

    def _forward(self, x, support_idx, query_idx):
        instance_embs = self.encoder(x)

        b, emb_dim, h, w = instance_embs.size()
        episode_size = b // (self.args.way * (self.args.shot+self.args.query) )

        support = instance_embs[support_idx.contiguous().view(-1)].unsqueeze(0)
        query   = instance_embs[query_idx.contiguous().view(-1)].unsqueeze(0)

        support = support.view(episode_size, self.args.shot, self.args.way, emb_dim, h, w)
        support = support.permute(0, 2, 1, 3, 4, 5)
        support = support.contiguous().view(episode_size, self.args.way*self.args.shot, emb_dim, h, w)

        logits = self.dn4_layer(query, support).view(episode_size*self.args.way*self.args.query, self.args.way) / self.args.temperature


        if self.training:
            if self.args.kd_loss == "KD":   # 传统的基于分类logits的蒸馏方法，返回学生和教师各自计算出的logits
                # （教师模型用线性分类头返回常规分类结果）
                teacher_logits = self.distill_layer(x)

                features = self.GAP(instance_embs)
                student_logits = self.fc(features.view(features.size(0), -1))
                
                return logits, student_logits, teacher_logits   # 三个返回值分别是：学生模型的小样本分类预测 + 学生模型的常规分类预测 + 教师模型的常规分类预测


                """ #(教师模型也按DN4方法返回小样本分类结果)已弃用
                instance_embs = self.distill_layer(x)

                if instance_embs is None:   # (仅当is_distill为false)
                    return logits, None

                b, emb_dim, h, w = instance_embs.size()
                episode_size = b // (self.args.way * (self.args.shot+self.args.query) )

                support = instance_embs[support_idx.contiguous().view(-1)].unsqueeze(0)
                query   = instance_embs[query_idx.contiguous().view(-1)].unsqueeze(0)

                support = support.view(episode_size, self.args.shot, self.args.way, emb_dim, h, w)
                support = support.permute(0, 2, 1, 3, 4, 5)
                support = support.contiguous().view(episode_size, self.args.way*self.args.shot, emb_dim, h, w)

                teacher_logits = self.dn4_layer(query, support).view(episode_size*self.args.way*self.args.query, self.args.way) / self.args.temperature

                return logits, teacher_logits
                """

            else:       # 其他蒸馏损失，一律返回学生和教师各自的中间编码（局部特征）
                student_encoding = instance_embs#.unsqueeze(0)
                teacher_encoding = self.distill_layer(x)#.unsqueeze(0)

                return logits, student_encoding, teacher_encoding
        else:
            return logits

        # forward函数有三种返回模式：
        # 1、验证测试时，只返回logits；
        # 2、训练阶段，基于logits的蒸馏，返回学生logits和教师logits；
        # 3、训练阶段，基于特征/关系的蒸馏，返回学生logtis和学生、教师分别所得的encodings(局部特征集合)



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
