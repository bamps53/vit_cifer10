import os
import argparse

import timm
from omegaconf import OmegaConf
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from ptbox import (
    MODELS, LOSSES, OPTIMIZERS, SCHEDULERS, METRICS,
    build_from_config,
    build_from_config_list, 
    build_from_config_dict
    )

from mixup import mixup_data, mixup_criterion
from initialize import initialize
initialize()

class LitModel(pl.LightningModule):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = build_from_config(cfg.model, MODELS)
        self.criterion = nn.CrossEntropyLoss()
        #san(sa_type=1, layers=(3, 3, 3, 3), kernels=[3, 3, 3, 3], num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.cfg.train.mixup:
            x, ya, yb, lam = mixup_data(x, y)
            logits = self(x)
            loss = mixup_criterion(self.criterion, logits, ya, yb, lam)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.log('train_loss', loss, prog_bar=True)
        if not self.cfg.train.mixup:
            acc = accuracy(preds, y)
            self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = build_from_config(cfg.optimizer, OPTIMIZERS, default_args={'params': self.model.parameters()})
        scheduler = build_from_config(cfg.scheduler, SCHEDULERS, default_args={'optimizer': optimizer})
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
        return trainloader

    def val_dataloader(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
        return testloader

    def test_dataloader(self):
        return self.val_dataloader()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str)
    parser.add_argument("--device_id", "-d", default="0", type=str)
    parser.add_argument("--start_fold", "-s", default=0, type=int)
    parser.add_argument("--end_fold", "-e", default=5, type=int)
    parser.add_argument("--debug", "-db", action="store_true")
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--inference", "-i", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    seed_everything(cfg.seed)
    model = LitModel(cfg)
    wandb_logger = WandbLogger(name=cfg.exp_id, project='vit_cifer10')
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.train.num_epochs, progress_bar_refresh_rate=20, precision=cfg.train.precision, logger=wandb_logger)
    trainer.fit(model)