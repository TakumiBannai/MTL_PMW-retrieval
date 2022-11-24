import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import stats
from torch.distributions import Beta
import warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CombinedLoss(nn.Module):
    def __init__(self, reg_loss, clsf_loss, alpha=10, size_average=True, strategy=None, class_weights=[0.05, 0.1, 0.15, 0.5, 0.2]):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.size_average = size_average
        self.reg_loss = reg_loss
        self.clsf_loss  = clsf_loss
        self.strategy = strategy
        self.class_weights = class_weights

    def forward(self, clsf, reg, targets, epoch, train_flag=False):
        if train_flag == False:
            loss_type = "mix"
        else:
            if self._is_clsf_epoch(epoch):
                batch_loss = self.clsf_loss(clsf, targets[:, 0].long())
                loss_type = "clsf"
            elif self._is_reg_epoch(epoch):
                batch_loss = self.reg_loss(reg, targets[:, 1].unsqueeze(1))
                loss_type = "reg"
            elif self._is_mix_epoch(epoch):
                batch_loss = self.clsf_loss(clsf, targets[:, 0].long()) + self.class_weights[4] * self.alpha * self.reg_loss(reg, targets[:, 1].unsqueeze(1))
                loss_type = "mix"
            else:
                raise RuntimeError("undefined epoch num!")
        return batch_loss, loss_type
    
    def _is_clsf_epoch(self,epoch):
        if epoch >= self.strategy["clsf_begin_epoch"] and epoch < self.strategy["clsf_end_epoch"]:
            return True
        else:
            return False

    def _is_reg_epoch(self,epoch):
        if epoch >= self.strategy["reg_begin_epoch"] and epoch < self.strategy["reg_end_epoch"]:
            return True
        else:
            return False

    def _is_mix_epoch(self,epoch):
        if epoch >= self.strategy["mix_begin_epoch"] and epoch < self.strategy["mix_end_epoch"]:
            return True
        else:
            return False
