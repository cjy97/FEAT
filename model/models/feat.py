import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel

from einops import rearrange
import copy

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class FEAT(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
        elif args.backbone_class == 'Res18':
            hdim = 512
        elif args.backbone_class == 'WRN':
            hdim = 640
        else:
            raise ValueError('')
        
        # self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)          
        # for k, v in self.slf_attn.named_parameters():
        #     v.requires_grad = False

        self.dn4_layer = DN4Layer(args.way, args.shot, args.query, n_k = 5)

        self.distill_layer = DistillLayer(
            self.encoder,
            args.encoder_path,
            args.is_distill,
        )
        # self.local_kd = Local_KD(args.way, args.shot+args.query)

    def _forward(self, x, support_idx, query_idx):

        # print("x.size(): ", x.size())   # [80, 3, 84, 84]
        instance_embs = self.encoder(x)

        # emb_dim = instance_embs.size(-1)
        b, emb_dim, h, w = instance_embs.size()
        episode_size = b // (self.args.way * (self.args.shot+self.args.query) )


        # organize support/query data
        # support = instance_embs[support_idx.contiguous().view(-1)]#.contiguous().view(*(support_idx.shape + (-1,)))
        # query   = instance_embs[query_idx.contiguous().view(-1)]#.contiguous().view(  *(query_idx.shape   + (-1,)))
        support = instance_embs[support_idx.contiguous().view(-1)].unsqueeze(0)
        query   = instance_embs[query_idx.contiguous().view(-1)].unsqueeze(0)

        support = support.view(episode_size, self.args.shot, self.args.way, emb_dim, h, w)
        support = support.permute(0, 2, 1, 3, 4, 5)
        support = support.contiguous().view(episode_size, self.args.way*self.args.shot, emb_dim, h, w)
        # print("support: ", support.size())  # [1, 5*1, 640, 5, 5]
        # print("query: ", query.size())      # [1, 75, 640, 5, 5]


        # support = support.permute(0, 1, 3, 4, 2)
        # print("support: ", support.size())  # [1, 5*1, 5, 5, 640]
        # support = support.contiguous().view(episode_size, (self.args.way*self.args.shot) * (h*w), emb_dim)
        # print("support: ", support.size())  # [1, 5*25, 640]
        # support = self.slf_attn(support, support, support)
        # support = support.view(episode_size, self.args.way*self.args.shot , h, w, emb_dim)
        # support = support.permute(0, 1, 4, 2, 3)
        # print("support: ", support.size())  # [1, 5*1, 640, 5, 5]

        logits = self.dn4_layer(query, support).view(episode_size*self.args.way*self.args.query, self.args.way) / self.args.temperature
        

        # # get mean of the support
        # proto = support.mean(dim=1) # Ntask x NK x d
        # print("proto: ", proto.size())      # [1, 5, 640]
        # num_batch = proto.shape[0]
        # num_proto = proto.shape[1]
        # num_query = np.prod(query_idx.shape[-2:])
        # print("num_query: ", num_query)     # 75
    
        # # query: (num_batch, num_query, num_proto, num_emb)
        # # proto: (num_batch, num_proto, num_emb)
        # proto = self.slf_attn(proto, proto, proto)
        # if self.args.use_euclidean:
        #     query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
        #     print("query: ", query.size())  # [75, 1, 640]
        #     proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        #     proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
        #     print("proto: ", proto.size())  # [75, 5, 640]

        #     print((proto - query).size())                   # [75, 5, 640]
        #     print(torch.sum((proto - query) ** 2, 2).size())# [75, 5]

        #     logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        #     print("logits: ", logits.size())# [75, 5]   全部5*15=75个query样本关于5个类别原型的距离
        # else:
        #     proto = F.normalize(proto, dim=-1) # normalize for cosine distance
        #     query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

        #     logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
        #     logits = logits.view(-1, num_proto)
        
        # for regularization
        if self.training:

            logits_reg = None
            
            # calc Local Distillation Loss while training
            student_feat = instance_embs
            teacher_feat = self.distill_layer(x)
            # print("student_feat: ", student_feat.size())    # [80, 640, 5, 5]
            # print("teacher_feat: ", teacher_feat.size())    # [80, 640, 5, 5]

            GAP = nn.AvgPool2d(5, stride=1)
            teacher_feat = GAP(teacher_feat).view(teacher_feat.size(0), -1).unsqueeze(0)   # [1, 80, 640]

            teacher_relation = torch.matmul(F.normalize(teacher_feat, p=2, dim=-1), 
                                            torch.transpose(F.normalize(teacher_feat, p=2, dim=-1), -1, -2))    # [1, 80, 80]
            
            student_feat = student_feat.permute(0, 2, 3, 1).contiguous().view(b, h*w, emb_dim).unsqueeze(0)     # [1, 80, 25, 640]


            s_relation = torch.matmul(F.normalize(student_feat.unsqueeze(2), p=2, dim=-1),            # [1, 80, 1, 25, 640]
                      torch.transpose(F.normalize(student_feat.unsqueeze(1), p=2, dim=-1), -1, -2))   # [1, 1, 80, 640, 25]
            
            # print("s_relation: ", s_relation.size())    # [1, 80, 80, 25, 25]

            top_k = 1
            topk_value, _ = torch.topk(s_relation, top_k, dim=-1)  # [1, 80, 80, 25, k]
            student_relation = torch.sum(topk_value, dim=[3, 4])      # [1, 80, 80]
            student_relation = student_relation / (top_k * h*w)
            student_relation = (student_relation + torch.transpose(student_relation, -1, -2) ) / 2

            # print("teacher_relation: ", teacher_relation)
            # print("student_relation: ", student_relation)

            
            criterion = nn.MSELoss(size_average=False, reduce=True)
            local_kd_loss = criterion(teacher_relation, student_relation)
            print("local_kd_loss: ", local_kd_loss)
            

            # print("training--")
            # aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
            #                       query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            # print("aux_task: ", aux_task.size())    # [1, 16, 5, 640]
            # num_query = np.prod(aux_task.shape[1:3])
            # print("num_query： ", num_query)        # 80
            # aux_task = aux_task.permute([0, 2, 1, 3])
            # print("aux_task: ", aux_task.size())    # [1, 5, 16, 640]
            # aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # print("aux_task: ", aux_task.size())    # [5, 16, 640]
            # # apply the transformation over the Aug Task
            # aux_emb = self.slf_attn(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            # print("aux_emb: ", aux_emb.size())      # [5, 16, 640]
            # # compute class mean
            # aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            # print("aux_emb: ", aux_emb.size())      # [1, 5, 16, 640]
            # aux_center = torch.mean(aux_emb, 2) # T x N x d
            # print("aux_center: ", aux_center.size())# [1, 5, 640]
            
            # if self.args.use_euclidean:
            #     aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            #     print("aux_task: ", aux_task.size())        # [80, 1, 640]
            #     aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            #     print("aux_center: ", aux_center.size())    # [1, 80, 5, 640]
            #     aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
            #     print("aux_center: ", aux_center.size())    # [80, 5, 640]
    
            #     logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            #     print("logits_reg: ", logits_reg.size())    # [80, 5]
            # else:
            #     aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
            #     aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
            #     logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
            #     logits_reg = logits_reg.view(-1, num_proto)            
            
            return logits, logits_reg, local_kd_loss
        else:
            return logits   

# class DistillKLLoss(nn.Module):
#     def __init__(self, T):
#         super(DistillKLLoss, self).__init__()
#         self.T = T

#     def forward(self, y_s, y_t):
#         if y_t is None:
#             return 0.0

#         p_s = F.log_softmax(y_s / self.T, dim=1)
#         p_t = F.softmax(y_t / self.T, dim=1)
#         loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.size(0)
#         return loss

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

class DistillLayer(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_path,
        is_distill,
    ):
        super(DistillLayer, self).__init__()
        self.encoder = self._load_state_dict(encoder, encoder_path, is_distill)

    def _load_state_dict(self, model, state_dict_path, is_distll):
        new_model = None
        if is_distll and state_dict_path is not None:
            new_model = copy.deepcopy(model)    # 先复制网络结构

            model_dict = new_model.state_dict()
            
            pretrained_dict = torch.load(state_dict_path)['params']
            pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items() if k.replace("encoder.", "") in model_dict}
            
            model_dict.update(pretrained_dict)
            new_model.load_state_dict(model_dict)   # 只将权重加载给教师模型

        return new_model

    @torch.no_grad()
    def forward(self, x):
        local_features = None
        if self.encoder is not None:
            local_features = self.encoder(x)        # 教师模型不含分类头，直接返回一组局部特征
        
        return local_features


# class Local_KD(nn.Module):
#     def __init__(self, way_num, shot_query):
#         super(Local_KD, self).__init__()

#         self.way_num = way_num
#         self.shot_query = shot_query
#         # self.n_k = n_k
#         # self.device = device
#         # self.normLayer = nn.BatchNorm1d(self.way_num * 2, affine=True)
#         # self.fcLayer = nn.Conv1d(1, 1, kernel_size=2, stride=1, dilation=5, bias=False)
    
#     # def _cal_cov_matrix_batch(self, feat):
#     #     e, _, n_local, c = feat.size()
#     #     feature_mean = torch.mean(feat, 2, True)
#     #     feat = feat - feature_mean
#     #     cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)
#     #     cov_matrix = torch.div(cov_matrix, n_local-1)
#     #     cov_matrix = cov_matrix + 0.01 * torch.eye(c).to(self.device)

#     #     return feature_mean, cov_matrix

#     def _cal_cov_batch(self, feat):
#         e, b, c, h, w = feat.size()
#         feat = feat.view(e, b, c, -1).permute(0, 1, 3, 2).contiguous()   # e, b, h*w, c
#         feat = feat.view(e, 1, b*h*w, c)                    # e, 1, b*h*w, c
#         feat_mean = torch.mean(feat, 2, True)               # e, 1, 1, c
#         feat = feat - feat_mean
#         cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)   # e, 1, c, c
#         cov_matrix = torch.div(cov_matrix, b*h*w-1) # b*h*w !!
#         cov_matrix = cov_matrix + 0.01 * torch.eye(c).cuda() #to(self.device)

#         return feat_mean, cov_matrix

#     def _calc_kl_dist_batch(self, mean1, cov1, mean2, cov2):
#         print("mean1: ", mean1.size())  # [e, 1, 1, 640]
#         print("cov1: ", cov1.size())    # [e, 1, 640, 640]
#         print("mean2: ", mean2.size())  # [e, 1, 1, 640]
#         print("cov2: ", cov2.size())    # [e, 1, 640, 640]

#         cov2_inverse = torch.inverse(cov2)
#         mean_diff = -(mean1 - mean2.squeeze(2).unsqueeze(1))

#         matrix_prod = torch.matmul(
#             cov1.unsqueeze(2), cov2_inverse.unsqueeze(1)
#         )
#         # print("matrix_prod: ", matrix_prod.size())

#         trace_dist = torch.diagonal(matrix_prod, offset=0, dim1=-2, dim2=-1)
#         trace_dist = torch.sum(trace_dist, dim=-1)
#         # print("trace_dist: ", trace_dist.size())

#         maha_prod = torch.matmul(
#             mean_diff.unsqueeze(3), cov2_inverse.unsqueeze(1)
#         )
#         maha_prod = torch.matmul(maha_prod, mean_diff.unsqueeze(4))
#         maha_prod = maha_prod.squeeze(4)
#         maha_prod = maha_prod.squeeze(3)

#         matrix_det = torch.slogdet(cov2).logabsdet.unsqueeze(1) - torch.slogdet(
#             cov1
#         ).logabsdet.unsqueeze(2)

#         kl_dist = trace_dist + maha_prod + matrix_det - mean1.size(3)

#         return kl_dist / 2.0


#     # 输入单个episode的所有样本（不区分support/query），返还局部蒸馏损失。
#     def forward(self, student_feat, teacher_feat):
#         e, b, c, h, w = student_feat.size()

#         student_mean, student_cov = self._cal_cov_batch(student_feat)
#         student_mean = student_mean.permute(1, 0, 2, 3)
#         student_cov = student_cov.permute(1, 0, 2, 3)
#         # student_feat = student_feat.view(e, b, c, -1).permute(0, 1, 3, 2).contiguous()

#         teacher_mean, teacher_cov = self._cal_cov_batch(teacher_feat)
#         teacher_mean = teacher_mean.permute(1, 0, 2, 3)
#         teacher_cov = teacher_cov.permute(1, 0, 2, 3)
#         # teacher_feat = teacher_feat.view(e, b, c, -1).permute(0, 1, 3, 2).contiguous()

#         kl_dis = self._calc_kl_dist_batch(student_mean, student_cov, teacher_mean, teacher_cov)
#         print("kl_dis: ", kl_dis.size())    # [e, 1, 1]
#         # print(kl_dis)

#         kl_mean = kl_dis.mean()
#         print("kl_mean: ", kl_mean)

#         return kl_mean
