import os

import numpy as np
import torch
import torch.optim as optim
import lightning.pytorch as pl
from lightning.pytorch import Callback
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr)
import wandb

import src.utils as utils

class PLModule(pl.LightningModule):
    def __init__(self, model, model_params, lr, scheduler, scheduler_params, margin=0.5,
                 neg_loss_start_epoch=10, init_ckpt=None):
        super(PLModule, self).__init__()
        self.model = utils.import_attr(model)(**model_params)
        self.lr = lr
        self.loss_fn = torch.nn.CosineEmbeddingLoss(margin=margin)
        self.emb = []
        self.emb_gt = []
        self.monitor = 'val/loss'
        self.monitor_mode = 'min'
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.neg_loss_start_epoch = neg_loss_start_epoch
        self.init_ckpt = init_ckpt

        if init_ckpt is not None:
            self.load_state_dict(torch.load(init_ckpt)['state_dict'])

    def forward(self, inputs):
        x = inputs['enrollments'].squeeze(1) # [B, C, T]
        return self.model(x)

    def _step(self, batch, batch_idx, mode):
        inputs, targets = batch
        emb = self(inputs)
        emb_gt = targets['embedding_gt'].squeeze(1)

        # Compute constrastive loss
        # +ve example
        loss = self.loss_fn(emb, emb_gt, torch.ones(emb.shape[0]).to(self.device))
        pos_loss = loss.item()
        # -ve examples
        if self.current_epoch >= self.neg_loss_start_epoch:
            for emb_neg in targets['embedding_neg']:
                emb_neg = emb_neg.squeeze(1)
                loss += self.loss_fn(emb, emb_neg, -torch.ones(emb.shape[0]).to(self.device))

        if batch_idx % 5 == 0 and (mode == 'val' or mode == 'test'):
            spk_ids = batch[0]['enrollments_id'][:, 0]
            for spk_id, _emb, _emb_gt in zip(spk_ids, emb, emb_gt):
                self.emb.append((spk_id, _emb))
                self.emb_gt.append((spk_id, _emb_gt))

        return emb, emb_gt, loss, pos_loss

    def training_step(self, batch, batch_idx):
        emb, emb_gt, loss, pos_loss = self._step(batch, batch_idx, 'train')
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=emb.shape[0])
        self.log('train/pos_loss', pos_loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=emb.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        emb, emb_gt, loss, pos_loss = self._step(batch, batch_idx, 'val')
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=emb.shape[0])
        self.log('val/pos_loss', pos_loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=emb.shape[0])
        return emb

    def test_step(self, batch, batch_idx):
        emb, emb_gt, loss, pos_loss = self._step(batch, batch_idx, 'test')
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=emb.shape[0])
        self.log('test/pos_loss', pos_loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=emb.shape[0])
        return emb

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler is not None:
            self.scheduler = utils.import_attr(self.scheduler)(optimizer, **self.scheduler_params)
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler_cfg = {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val/loss",
                    "strict": False,
                    "name": "reduce_on_plateau_lrsched"
                }
            else:
                scheduler_cfg = self.scheduler
            return [optimizer], [scheduler_cfg]
        else:
            return optimizer

class Logger(Callback):
    def _log_embeddings(self, logger, key, samples):
        embed_size = samples[0][1].shape[-1]
        table = wandb.Table(columns=['speaker_id'] + [f'embed_{i}' for i in range(embed_size)])
        for sample in samples:
            row = [sample[0]] + sample[1].cpu().tolist()
            table.add_data(*row)
        logger.experiment.log({key: table})

    def on_validation_end(self, trainer, pl_module):
        if pl_module.current_epoch == 0:
            self._log_embeddings(
                trainer.logger, "val/embeddings_gt", pl_module.emb_gt)
        self._log_embeddings(
            trainer.logger, "val/embeddings", pl_module.emb)
        pl_module.emb.clear()
        pl_module.emb_gt.clear()
