import torch
import torch.optim.lr_scheduler as lrs
from torch.optim.optimizer import Optimizer

import logging

class Linear_schedular_with_warmup(lrs.LRScheduler):
    def __init__(self, optimizer, 
                 warmup_start_factor=0.001,
                 start_factor=1.0,
                 end_factor=0.01,
                 warmup_iters=5,
                 total_iters=10, 
                 last_epoch=-1, 
                 verbose=False):
        self.warmup_start_factor = warmup_start_factor
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        super(Linear_schedular_with_warmup, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return self._get_closed_form_lr()

        current_factor = 1.0
        if self.last_epoch < self.warmup_iters and self.last_epoch >= 0:
            current_factor = (1. + (self.start_factor - self.warmup_start_factor) / (self.warmup_iters * self.warmup_start_factor + (self.last_epoch - 1) * (self.start_factor - self.warmup_start_factor)))
        elif self.last_epoch <= self.total_iters and self.last_epoch > 0:
            current_factor = (1. + (self.end_factor - self.start_factor) / ((self.total_iters-self.warmup_start_factor) * self.start_factor + ((self.last_epoch-self.warmup_iters) - 1) * (self.end_factor - self.start_factor)))

        print(f"current_factor: {current_factor}")

        return [group['lr'] * current_factor for group in self.optimizer.param_groups]


    def _get_closed_form_lr(self):
        current_factor = self.end_factor
        if self.last_epoch < self.warmup_iters:
            current_factor = self.warmup_start_factor + \
                (self.start_factor - self.warmup_start_factor) * self.last_epoch / self.warmup_iters
        elif self.last_epoch <= self.total_iters:
            current_factor = self.start_factor + \
                (self.end_factor - self.start_factor) * \
                    (self.last_epoch - self.warmup_iters) / \
                        (self.total_iters - self.warmup_iters)
        
        return [base_lr * current_factor for base_lr in self.base_lrs]
    




class Linear_schedular_seg(lrs.LRScheduler):
    def __init__(self, optimizer, 
                 seg_list: list, # [[start_factor, end_factor, period_iters], ...]
                 last_epoch=-1, 
                 verbose=False):
        self.seg_list = seg_list
        logging.info(f"Using Linear_schedular_seg as the Learning Rate Scheduler: seg_list: {seg_list}")
        self.landmarks = []
        seg_sum = 0
        for seg in seg_list:
            seg_sum += seg[2]
            self.landmarks.append(seg_sum)
        logging.info(f"Linear_schedular_seg: landmarks: {self.landmarks}")
        self.final_factor = seg_list[-1][1]
        super(Linear_schedular_seg, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return self._get_closed_form_lr()


    def _get_closed_form_lr(self):
        current_factor = self.final_factor
        current_dist_beyond_landmark = self.last_epoch
        for idx, seg in enumerate(self.seg_list):
            if self.last_epoch < self.landmarks[idx]:
                current_factor = seg[0] + (seg[1] - seg[0]) * current_dist_beyond_landmark / seg[2]
                break
            current_dist_beyond_landmark -= seg[2]
        
        return [base_lr * current_factor for base_lr in self.base_lrs]
    