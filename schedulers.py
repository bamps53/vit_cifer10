import math

from torch.optim.lr_scheduler import _LRScheduler
import ptbox


@ptbox.SCHEDULERS.register
class HalfCosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max, range=2, last_epoch=-1):
        self.T_max = T_max
        self.range = range
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % (2 * self.T_max) < self.T_max:
            cos_unit = 0.5 * (math.cos(math.pi * self.last_epoch / self.T_max) - 1)
        else:
            cos_unit = 0.5 * (math.cos(math.pi * (self.last_epoch / self.T_max - 1)) - 1)

        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * 10 ** (-self.range)
            range = math.log10(base_lr - math.log10(min_lr))
            lrs.append(10 ** (math.log10(base_lr) + range * cos_unit))
        return lrs