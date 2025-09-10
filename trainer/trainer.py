# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from base import BaseTrainer
from torch.nn.utils import clip_grad_norm_
from utils import MetricTracker, inf_loop
from utils import MultiLossMetricTracker
from sklearn.metrics import classification_report
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, test_data_loader=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # Replace train_metrics with MetricTracker for individual metrics
        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        self.model.train()
        self.train_metrics.reset()
        
        # Add progress bar
        pbar = tqdm(self.data_loader, desc=f'Epoch {epoch}', leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)  # [out1, out2, out3] (upsampled)
            outs = outputs  # raw outputs for loss
            softmaxed = [torch.nn.functional.softmax(o, dim=1) for o in outputs]  # softmaxed outputs for prediction/metrics
            # print("--------------------------------")
            # print("Train Epoch")
            # print("--------------------------------")
            # Use the highest resolution output for prediction
            preds = [torch.argmax(sm, dim=1) for sm in softmaxed]
            loss_target = target.clone()
            loss_target[loss_target != 0] -= 1
            loss_target = loss_target.squeeze(1)
            # Downsample loss_target for each prediction scale
            loss_targets = [
                loss_target,
                torch.nn.functional.interpolate(loss_target.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').long().squeeze(1),
                torch.nn.functional.interpolate(loss_target.unsqueeze(1).float(), scale_factor=0.125, mode='nearest').long().squeeze(1)
            ]
            # Compute individual losses for each prediction/target for logging
            individual_losses = []
            ce_loss = torch.nn.CrossEntropyLoss()
            for i, (out, t) in enumerate(zip(outs, loss_targets)):
                # If target has shape (B, 1, H, W), squeeze to (B, H, W)
                if t.dim() == 4 and t.size(1) == 1:
                    t = t.squeeze(1)
                # Resize target if needed (shouldn't be needed if targets are preprocessed, but keep for safety)
                if out.shape[2:] != t.shape[1:]:
                    t = torch.nn.functional.interpolate(t.unsqueeze(1).float(), size=out.shape[2:], mode='nearest').long().squeeze(1)
                l = ce_loss(out, t)
                individual_losses.append(l.item() if hasattr(l, 'item') else float(l))
            # Use the new criterion interface
            loss = self.criterion(outs, loss_targets)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 0.05)
            self.optimizer.step()

            # print(f"type loss_target: {type(loss_target)}")
            # print(f"shape of loss_target: {loss_target.shape}")
            # Update MetricTracker for training
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                # print(f"Type of accuracy: {type(met(softmaxed, loss_target))}")
                # print(met(softmaxed, loss_target))
                # print(met.__name__)
                self.train_metrics.update(
                    met.__name__, met(softmaxed, loss_target))
            # print("---SUCCESSFUL---")
            # import sys; sys.exit(0)

            # Update progress bar with current loss
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(self.valid_data_loader)
            log.update(**{'val_'+k: v for k, v in val_log.items()})
        if self.do_test and epoch == self.config['trainer']['epochs']:
            try:
                best_path = str(self.checkpoint_dir / 'model_best.pth')
                self._resume_checkpoint(best_path)
                self.logger.info("Testing current best: model_best.pth ...")
                test_log = self._valid_epoch(self.test_data_loader)
                log.update(**{'test_'+k: v for k, v in test_log.items()})
            except FileNotFoundError:
                self.logger.info("No model_best.pth found, skipping testing ...")

        # print("---SUCCESSFUL---")
        # import sys; sys.exit(0)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, data_loader):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        preds = np.array([])
        targets = np.array([])

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            # Add progress bar for validation
            pbar = tqdm(data_loader, desc='Validation', leave=False)
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)  # In eval mode, this is [out1] (list of length 1)
                softmaxed = [torch.nn.functional.softmax(o, dim=1) for o in outputs]
                pred = torch.argmax(softmaxed[0], dim=1)  # Use the first (and only) output
                loss_target = target.clone()
                loss_target[loss_target != 0] -= 1
                loss_target = loss_target.squeeze(1)
                # In eval mode, use regular CrossEntropyLoss since we only have one output
                ce_loss = torch.nn.CrossEntropyLoss()
                loss = ce_loss(outputs[0], loss_target)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(softmaxed[0], loss_target))

                label_valid_indices = (target.view(-1) != 0)
                valid_pred = pred.view(-1)[label_valid_indices]
                valid_label = target.view(-1)[label_valid_indices] - 1
                preds = np.concatenate((preds, valid_pred.view(-1).cpu()), axis=0)
                targets = np.concatenate((targets, valid_label.view(-1).cpu()), axis=0)
                
                # Update validation progress bar
                pbar.set_postfix({'Val Loss': f'{loss.item():.6f}'})
        log = self.valid_metrics.result()
        log['classification_report'] = "\n" + classification_report(targets, preds, target_names=('Non-Forest', 'Forest'))
        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
