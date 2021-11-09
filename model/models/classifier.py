import torch
import torch.nn as nn
import numpy as np
from model.utils import euclidean_metric
import torch.nn.functional as F

import copy

class DistillLayer(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_path,
    ):
        super(DistillLayer, self).__init__()
        # 教师模型只有encoder部分
        self.encoder = self._load_state_dict(encoder, encoder_path)
    
    def _load_state_dict(self, model, state_dict_path):
        new_model = None
        if state_dict_path is not None:
            new_model = copy.deepcopy(model)    # 先从学生模型加载网络结构

            model_dict = new_model.state_dict()
            
            pretrained_dict = torch.load(state_dict_path)['params'] # 从预训练读取权重
            pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items() if k.replace("encoder.", "") in model_dict}
            
            model_dict.update(pretrained_dict)
            new_model.load_state_dict(model_dict)   # 只将预训练权重加载给教师模型

        return new_model
    
    @torch.no_grad()
    def forward(self, x):
        local_features = None
        if self.encoder is not None:
            local_features = self.encoder(x)
        
        return local_features

    
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

        self.distill_layer = DistillLayer(
            self.encoder,
            args.encoder_path,
        )
        self.local_kd = Local_KD()
        self.avgpool = nn.AvgPool2d(5, stride=1) # 手动加上GAP层，供fc计算分类预测

    def forward(self, data, is_emb = False):
        out = self.encoder(data)                # 由于res12中原有的GAP被注释掉了，这里的out和distill_out都是局部特征

        print("out: ", out.size())
        distill_output = self.distill_layer(data)
        student_feat = out.unsqueeze(0)
        teacher_feat = distill_output.unsqueeze(0)
        print("student_feat: ", student_feat)    # [1, 80, 640, 5, 5]
        # print("teacher_feat: ", teacher_feat)    # [1, 80, 640, 5, 5]
        local_kd_loss = self.local_kd(student_feat, teacher_feat)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        if not is_emb:
            out = self.fc(out)
        
        return out, local_kd_loss
    
    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        query = self.encoder(data_query)
        
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim

class Local_KD(nn.Module):
    def __init__(self, way_num, shot_query):
        super(Local_KD, self).__init__()

        self.way_num = way_num
        self.shot_query = shot_query
        # self.n_k = n_k
        # self.device = device
        # self.normLayer = nn.BatchNorm1d(self.way_num * 2, affine=True)
        # self.fcLayer = nn.Conv1d(1, 1, kernel_size=2, stride=1, dilation=5, bias=False)
    
    # def _cal_cov_matrix_batch(self, feat):
    #     e, _, n_local, c = feat.size()
    #     feature_mean = torch.mean(feat, 2, True)
    #     feat = feat - feature_mean
    #     cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)
    #     cov_matrix = torch.div(cov_matrix, n_local-1)
    #     cov_matrix = cov_matrix + 0.01 * torch.eye(c).to(self.device)

    #     return feature_mean, cov_matrix

    def _cal_cov_batch(self, feat):
        e, b, c, h, w = feat.size()
        feat = feat.view(e, b, c, -1).permute(0, 1, 3, 2).contiguous()   # e, b, h*w, c
        feat = feat.view(e, 1, b*h*w, c)                    # e, 1, b*h*w, c
        feat_mean = torch.mean(feat, 2, True)               # e, 1, 1, c
        feat = feat - feat_mean
        cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)   # e, 1, c, c
        cov_matrix = torch.div(cov_matrix, b*h*w-1) # b*h*w !!
        cov_matrix = cov_matrix + 0.01 * torch.eye(c).cuda() #to(self.device)

        return feat_mean, cov_matrix

    def _calc_kl_dist_batch(self, mean1, cov1, mean2, cov2):
        print("mean1: ", mean1.size())  # [e, 1, 1, 640]
        print("cov1: ", cov1.size())    # [e, 1, 640, 640]
        print("mean2: ", mean2.size())  # [e, 1, 1, 640]
        print("cov2: ", cov2.size())    # [e, 1, 640, 640]

        cov2_inverse = torch.inverse(cov2)
        mean_diff = -(mean1 - mean2.squeeze(2).unsqueeze(1))

        matrix_prod = torch.matmul(
            cov1.unsqueeze(2), cov2_inverse.unsqueeze(1)
        )
        # print("matrix_prod: ", matrix_prod.size())

        trace_dist = torch.diagonal(matrix_prod, offset=0, dim1=-2, dim2=-1)
        trace_dist = torch.sum(trace_dist, dim=-1)
        # print("trace_dist: ", trace_dist.size())

        maha_prod = torch.matmul(
            mean_diff.unsqueeze(3), cov2_inverse.unsqueeze(1)
        )
        maha_prod = torch.matmul(maha_prod, mean_diff.unsqueeze(4))
        maha_prod = maha_prod.squeeze(4)
        maha_prod = maha_prod.squeeze(3)

        matrix_det = torch.slogdet(cov2).logabsdet.unsqueeze(1) - torch.slogdet(
            cov1
        ).logabsdet.unsqueeze(2)

        kl_dist = trace_dist + maha_prod + matrix_det - mean1.size(3)

        return kl_dist / 2.0


    # 输入单个episode的所有样本（不区分support/query），返还局部蒸馏损失。
    def forward(self, student_feat, teacher_feat):
        e, b, c, h, w = student_feat.size()

        student_mean, student_cov = self._cal_cov_batch(student_feat)
        student_mean = student_mean.permute(1, 0, 2, 3)
        student_cov = student_cov.permute(1, 0, 2, 3)
        # student_feat = student_feat.view(e, b, c, -1).permute(0, 1, 3, 2).contiguous()

        teacher_mean, teacher_cov = self._cal_cov_batch(teacher_feat)
        teacher_mean = teacher_mean.permute(1, 0, 2, 3)
        teacher_cov = teacher_cov.permute(1, 0, 2, 3)
        # teacher_feat = teacher_feat.view(e, b, c, -1).permute(0, 1, 3, 2).contiguous()

        kl_dis = self._calc_kl_dist_batch(student_mean, student_cov, teacher_mean, teacher_cov)
        print("kl_dis: ", kl_dis.size())    # [e, 1, 1]
        # print(kl_dis)

        kl_mean = kl_dis.mean()
        print("kl_mean: ", kl_mean)

        return kl_mean