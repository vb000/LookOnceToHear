import os

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from lightning.pytorch import Callback
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr)
import wandb
from src.eval.binaural import ild_diff, itd_diff
import torch

from src.losses.LossFn import LossFn

import src.utils as utils

class PLModule(pl.LightningModule):
    def __init__(self, model, model_params, lr, init_ckpt=None,
                 dir_loss=False, embed_model=None, embed_model_params=None,
                 scheduler=None, scheduler_params=None):
        
        super(PLModule, self).__init__()
        self.model = utils.import_attr(model)(**model_params)
        if embed_model is not None:
            self.embed_model = utils.import_attr(embed_model)(**embed_model_params)
        self.lr = lr
        self.dir_loss = None
        if dir_loss:
            self.dir_loss = nn.CrossEntropyLoss()

        # Metric to monitor
        self.monitor = 'val/si_snr_i'
        self.monitor_mode = 'max'

        # Initaize weights if checkpoint is provided
        if init_ckpt is not None:
            self.load_state_dict(torch.load(init_ckpt)['state_dict'])

        self.log_samples = []
        
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    def forward(self, x, embedding):
        return self.model(x, embedding)

    def loss_fn(self, pred, tgt):
        return -snr(pred, tgt)

    def _metric_i(self, metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append((metric(p, t) - metric(s, t)).mean())
        return torch.stack(_vals)

    def _step(self, batch, batch_idx, step='train'):
        inputs, targets = batch
        batch_size = inputs['mixture'].shape[0]

        # Forward pass
        if self.dir_loss is None:
            output = self.model(inputs['mixture'], targets['embedding_gt'])
        else:
            output, dir = self.model(inputs['mixture'], targets['embedding_gt'], dir=True)
        embed = targets['embedding_gt'].squeeze(1)

        # Compute loss
        loss = self.loss_fn(output, targets['target']).mean()
        if self.dir_loss is not None:
            gt_shifts = inputs['tgt_shift']
            gt_indices = self.model._shifts_to_indices(gt_shifts)
            dir = dir.mean(1)
            dir_loss = self.dir_loss(dir, gt_indices)
            loss += dir_loss

            # Log direction accuracy
            dir_pred = dir.argmax(1) * (180.0 / dir.shape[1])
            gt_angles = gt_indices * (180.0 / dir.shape[1])
            dir_error = torch.abs(dir_pred - gt_angles).mean()
            self.log(
                f'{step}/dir_error', dir_error, batch_size=batch_size, on_step=False,
                on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(
                f'{step}/dir_loss', dir_loss, batch_size=batch_size, on_step=False,
                on_epoch=True, prog_bar=False, sync_dist=True)

        # Use first and last channels (l/r) to compute metrics
        inputs['mixture'] = inputs['mixture'][:, [0,-1]]
            
        # Log metrics
        snr_i = self._metric_i(snr, inputs['mixture'], output, targets['target'])
        si_snr_i = self._metric_i(si_snr, inputs['mixture'], output, targets['target'])

        on_step = step == 'train'
        self.log(
            f'{step}/loss', loss, batch_size=batch_size, on_step=on_step,
            on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            f'{step}/snr_i', snr_i.mean(),
            batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False,
            sync_dist=True)
        self.log(
            f'{step}/si_snr_i', si_snr_i.mean(),
            batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True,
            sync_dist=True)

        if batch_idx % 5 == 0 and (step == 'val' or step == 'test'):
            spk_ids = batch[0]['enrollments_id'][:, 0]
            for spk_id, emb in zip(spk_ids, embed):
                self.log_samples.append((spk_id, emb))

        return loss, output, embed

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, batch_idx, step='train')
        return loss

    def validation_step(self, batch, batch_idx):
        _, output, embed = self._step(batch, batch_idx, step='val')
        return output, embed

    def test_step(self, batch, batch_idx):
        _, output, embed = self._step(batch, batch_idx, step='test')
        return output, embed

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
        self._log_embeddings(
            trainer.logger, "val/embeddings", pl_module.log_samples)
        pl_module.log_samples.clear()
