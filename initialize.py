from ptbox.registry import Registry
import torch
import albumentations

MODELS = Registry('models')
LOSSES = Registry('losses')
OPTIMIZERS = Registry('optimizers')
SCHEDULERS = Registry('schedulers')
DATASETS = Registry('datasets')
CALLBACKS = Registry('callbacks')
METRICS = Registry('metrics')
TRANSFORMS = Registry('transforms')

# losses
for k, v in torch.nn.__dict__.items():
    if 'Loss' in k:
        if callable(v) and isinstance(v, type):
            LOSSES.register(v)

# optimizers
for k, v in torch.optim.__dict__.items():
    if callable(v):
        OPTIMIZERS.register(v)

# schedulers
for k, v in torch.optim.lr_scheduler.__dict__.items():
    if callable(v):
        SCHEDULERS.register(v)

# transforms
for k,v in albumentations.__dict__.items():
    if callable(v) and isinstance(v, type):
        TRANSFORMS.register(v)

def initialize():
    from models import san
    from schedulers import HalfCosineAnnealingLR
