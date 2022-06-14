import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm

# local_kd = Local_KD(args.way, args.shot+args.query)
mse_loss = nn.MSELoss(size_average=False, reduce=True)
GAP = nn.AvgPool2d(5, stride=1)

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


class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux
    
    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        # start FSL training
        label, label_aux = self.prepare_label()
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()

            correct = 0 #
            total = 0   #
            sum_ce_loss = 0.0
            sum_kd_loss = 0.0
            for batch in self.train_loader:
                self.train_step += 1

                if torch.cuda.is_available():
                    data, gt_label = [_.cuda() for _ in batch]
                else:
                    data, gt_label = batch[0], batch[1]
               
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                # logits, reg_logits = self.para_model(data)
                # if reg_logits is not None:
                #     loss = F.cross_entropy(logits, label)
                #     total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
                # else:
                #     loss = F.cross_entropy(logits, label)
                #     total_loss = F.cross_entropy(logits, label)

                results = self.para_model(data)
                # print("results: ", len(results))
                if args.kd_loss == "KD":
                    logits, student_logits, teacher_logits = results

                    # print("student_logits: ", student_logits.size())
                    # print("teacher_logits: ", teacher_logits.size())
                    if teacher_logits is not None:
                        T = 4.0
                        p_s = F.log_softmax(student_logits / T, dim=1)
                        p_t = F.softmax(teacher_logits)
                        # print("p_s: ", p_s)
                        # print("p_t: ", p_t)
                        kd_loss = F.kl_div(
                            p_s,
                            p_t,
                            reduction='sum'
                            # size_average=False
                        ) #* (T**2)
                    else:
                        kd_loss = 0.0
                elif args.kd_loss == "global_KD":
                    logits, student_encoding, teacher_encoding = results
                    # print("student_encoding: ", student_encoding.size())
                    # print("teacher_encoding: ", teacher_encoding.size())
                    if teacher_encoding is not None:
                        student_feat = student_encoding
                        teacher_feat = teacher_encoding

                        student_feat = GAP(student_encoding).view(student_feat.size(0), -1).unsqueeze(0)
                        teacher_feat = GAP(teacher_encoding).view(teacher_feat.size(0), -1).unsqueeze(0)
                        
                        print("student_feat: ", student_feat.size())
                        print("teacher_feat: ", teacher_feat.size())

                        T = 4.0
                        p_s = F.log_softmax(student_feat / T, dim=1)
                        p_t = F.softmax(teacher_feat)
                        kd_loss = F.kl_div(
                            p_s,
                            p_t,
                            # reduction='sum'
                            size_average=True
                        ) #* (T**2)

                        # dim不一致如何解决？

                    else:
                        kd_loss = 0.0
                elif args.kd_loss == "relation_KD":
                    logits, student_encoding, teacher_encoding = results

                    if teacher_encoding is not None:
                        student_feat = student_encoding
                        teacher_feat = teacher_encoding

                        student_feat = GAP(student_encoding).view(student_feat.size(0), -1).unsqueeze(0)
                        teacher_feat = GAP(teacher_encoding).view(teacher_feat.size(0), -1).unsqueeze(0)

                        teacher_relation = torch.matmul(F.normalize(teacher_feat, p=2, dim=-1),
                                            torch.transpose(F.normalize(teacher_feat, p=2, dim=-1), -1, -2))
                        student_relation = torch.matmul(F.normalize(student_feat, p=2, dim=-1),
                                            torch.transpose(F.normalize(student_feat, p=2, dim=-1), -1, -2))
                        
                        kd_loss = mse_loss(teacher_relation, student_relation)

                    else:
                        kd_loss = 0.0
                elif args.kd_loss == "local_KD":
                    logits, student_encoding, teacher_encoding = results

                    student_feat = student_encoding.unsqueeze(0)
                    teacher_feat = teacher_encoding.unsqueeze(0)

                    local_kd = Local_KD(args.way, args.shot+args.query)
                    kd_loss = local_kd(student_feat, teacher_feat)
                
                elif args.kd_loss == "local_KD_pos":
                    logits, student_encoding, teacher_encoding = results

                    student_feat = student_encoding
                    teacher_feat = teacher_encoding
                    # print("student_feat: ", student_feat.size())    # [80, 640, 5, 5]
                    # print("teacher_feat: ", teacher_feat.size())
                    # 这里本来应该将形如[bs, emb_dim, h, w]的feat数据变形成[bs*h*w, emb_dim]的形式，但经测试不转换其实对结果没有影响

                    T = 4.0
                    p_s = F.log_softmax(student_feat / T, dim=1)
                    p_t = F.softmax(teacher_feat / T, dim=1)
                    kd_loss = F.kl_div(
                        p_s,
                        p_t,
                        size_average=False
                    ) * (T**2)
                
                elif args.kd_loss == "local_KD_rel":
                    logits, student_encoding, teacher_encoding = results

                    b, emb_dim, h, w = student_encoding.size()

                    student_feat = student_encoding
                    teacher_feat = teacher_encoding

                    student_feat = student_feat.permute(0, 2, 3, 1).contiguous().view(b, h*w, emb_dim)
                    teacher_feat = teacher_feat.permute(0, 2, 3, 1).contiguous().view(b, h*w, emb_dim)

                    student_relation = torch.matmul(F.normalize(student_feat, p=2, dim=-1), 
                                            torch.transpose(F.normalize(student_feat, p=2, dim=-1), -1, -2))

                    teacher_relation = torch.matmul(F.normalize(teacher_feat, p=2, dim=-1), 
                                            torch.transpose(F.normalize(teacher_feat, p=2, dim=-1), -1, -2))
                    # print("teacher_relation matrix: ", teacher_relation.size()) # [80, 25, 25]
                    # print("teacher_relation matrix: ", teacher_relation)

                    # criterion = nn.L1Loss(size_average=False, reduce=True)
                    criterion = nn.MSELoss(size_average=False, reduce=True)
                    kd_loss = criterion(teacher_relation, student_relation)

                elif args.kd_loss == "ALL":
                    logits, student_logits, teacher_logits, student_encoding, teacher_encoding = results
                    b, emb_dim, h, w = student_encoding.size()


                    T = 4.0
                    p_s = F.log_softmax(student_logits / T, dim=1)
                    p_t = F.softmax(teacher_logits)
                    loss_KD = F.kl_div(
                        p_s,
                        p_t,
                        reduction='sum'
                        # size_average=False
                    ) #* (T**2)
                    

                    student_feat = student_encoding
                    teacher_feat = teacher_encoding
                    T = 4.0
                    p_s = F.log_softmax(student_feat / T, dim=1)
                    p_t = F.softmax(teacher_feat / T, dim=1)
                    loss_local_kd_pos = F.kl_div(
                        p_s,
                        p_t,
                        size_average=False
                    ) * (T**2)


                    student_feat = student_encoding
                    teacher_feat = teacher_encoding
                    student_feat = student_feat.permute(0, 2, 3, 1).contiguous().view(b, h*w, emb_dim)
                    teacher_feat = teacher_feat.permute(0, 2, 3, 1).contiguous().view(b, h*w, emb_dim)
                    student_relation = torch.matmul(F.normalize(student_feat, p=2, dim=-1), 
                                            torch.transpose(F.normalize(student_feat, p=2, dim=-1), -1, -2))
                    teacher_relation = torch.matmul(F.normalize(teacher_feat, p=2, dim=-1), 
                                            torch.transpose(F.normalize(teacher_feat, p=2, dim=-1), -1, -2))
                    criterion = nn.MSELoss(size_average=False, reduce=True)
                    loss_local_kd_rel = criterion(teacher_relation, student_relation)

                    kd_loss = loss_KD*1.0 + loss_local_kd_pos*1.0 + loss_local_kd_rel*1.0


                else:
                    logits = results[0]
                    kd_loss = 0.0

                ce_loss = F.cross_entropy(logits, label)
                kd_loss = kd_loss * args.kd_weight
                print("ce_loss: ", ce_loss)
                print("kd_loss: ", kd_loss)
                total_loss = ce_loss + kd_loss

                sum_ce_loss += ce_loss
                sum_kd_loss += kd_loss
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted==label).sum().item()
                total += label.size()[0]
                    
                tl2.add(ce_loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)    

                # refresh start_tm
                start_tm = time.time()

            self.lr_scheduler.step()
            vl, va, vap = self.try_evaluate(epoch)
            self.epoch_record(epoch, vl, va, vap, train_acc = correct/total, avg_ce_loss = sum_ce_loss / len(self.train_loader), avg_kd_loss = sum_kd_loss / len(self.train_loader))

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((10000, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        return vl, va, vap
    
    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))            

    def epoch_record(self, epoch, vl, va, vap, train_acc, avg_ce_loss, avg_kd_loss):
        print(self.args.save_path)
        with open(osp.join(self.args.save_path, 'record.txt'), 'a') as f:
            f.write('epoch {}: train_acc={:.4f}, eval_loss={:.4f}, eval_acc={:.4f}+{:.4f}, avg_ce_loss={:.4f}, avg_kd_loss={:.4f}\n'.format(epoch, train_acc, vl, va, vap, avg_ce_loss, avg_kd_loss))