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
                    logits, teacher_logits = results

                    # print("student_logits: ", logits.size())
                    # print("teacher_logits: ", teacher_logits.size())
                    if teacher_logits is not None:
                        T = 4.0
                        p_s = F.log_softmax(logits / T, dim=1)
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