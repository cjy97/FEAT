import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel
from model.models.DistillLayer import DistillLayer
from model.models.Prune import prune_resnet

# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)


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

        features = self.GAP(instance_embs)
        features = features.view(features.size(0), -1)

        emb_dim = features.size(-1)

        # organize support/query data
        support = features[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query   = features[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if True: # self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
            proto = proto.contiguous().view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else: # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        if self.training:
            if self.args.kd_loss == "KD":   # 传统的基于分类logits的蒸馏方法，返回学生和教师各自计算出的logits
                # （教师模型用线性分类头返回常规分类结果）
                teacher_logits = self.distill_layer(x)

                features = self.GAP(instance_embs)
                student_logits = self.fc(features.view(features.size(0), -1))
                
                return logits, student_logits, teacher_logits   # 三个返回值分别是：学生模型的小样本分类预测 + 学生模型的常规分类预测 + 教师模型的常规分类预测
            else:       # 其他蒸馏损失，一律返回学生和教师各自的中间编码（局部特征）
                student_encoding = instance_embs#.unsqueeze(0)
                teacher_encoding = self.distill_layer(x)#.unsqueeze(0)

                return logits, student_encoding, teacher_encoding
        else:
            return logits
