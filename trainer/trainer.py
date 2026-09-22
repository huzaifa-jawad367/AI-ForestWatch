# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from base import BaseTrainer
from model.segmentation_metrics import MaskedSegmentationMetrics
from torch.nn.utils import clip_grad_norm_
from utils import (
    MetricTracker,
    create_grad_scaler,
    encode_segmentation_target,
    inf_loop,
    resolve_precision,
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import copy
import time
from datetime import datetime


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    @staticmethod
    def _loss_input(model, logits, probabilities):
        # Every exported architecture returns (logits, probabilities).
        # FocalLoss2d performs log_softmax internally; never feed probabilities.
        return logits

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, test_data_loader=None,
                 lr_scheduler=None, len_epoch=None):
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.precision = resolve_precision(config['trainer'], device)
        self.use_amp = self.precision.use_autocast
        self.amp_dtype = self.precision.autocast_dtype
        self.scaler = create_grad_scaler(self.precision)
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        trainer_config = config['trainer']
        self.mask_unknown_labels = bool(
            trainer_config.get('mask_unknown_labels', False)
        )
        self.unknown_label = int(trainer_config.get('unknown_label', 0))
        self.ignore_index = int(trainer_config.get('ignore_index', -100))
        self.boundary_dilation_ratio = float(
            trainer_config.get('boundary_dilation_ratio', 0.02)
        )
        if self.mask_unknown_labels and self.ignore_index != -100:
            raise ValueError(
                "The corrected masking protocol currently requires ignore_index=-100"
            )
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
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss')
        self.valid_metrics = MetricTracker('loss')
        
        # Initialize TensorBoard writer
        tb_dir = os.path.join(config.models_dir, 'tensorboard_logs')
        os.makedirs(tb_dir, exist_ok=True)
        tb_kwargs = {'log_dir': tb_dir}
        if config.resume is not None:
            # Hide any partial/crashed events at or after the resumed epoch while
            # retaining the existing event history in this same log directory.
            tb_kwargs['purge_step'] = self.start_epoch
        self.tb_writer = SummaryWriter(**tb_kwargs)


        # Initialize persistent log file
        self._init_log_file()

    def _prepare_target(self, target, num_classes):
        return encode_segmentation_target(
            target,
            mask_unknown_labels=self.mask_unknown_labels,
            unknown_label=self.unknown_label,
            ignore_index=self.ignore_index,
            num_classes=num_classes,
        )

    def _supervised_pixel_count(self, target, labelled_mask):
        if self.mask_unknown_labels:
            return int(labelled_mask.sum().item())
        return target.numel()

    def _prepare_metric_target(self, target, num_classes):
        # Reporting is always based only on labelled pixels, including when an
        # old protocol deliberately retains its historical loss semantics.
        return encode_segmentation_target(
            target,
            mask_unknown_labels=True,
            unknown_label=self.unknown_label,
            ignore_index=self.ignore_index,
            num_classes=num_classes,
        )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        epoch_started = time.perf_counter()
        skipped_optimizer_steps = 0
        all_ignored_batches = 0
        labelled_pixels = 0
        ignored_pixels = 0
        global_metrics = MaskedSegmentationMetrics(
            ignore_index=self.ignore_index,
            boundary_dilation_ratio=self.boundary_dilation_ratio,
            compute_boundary=False,
        )
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        
        # Create progress bar for training
        pbar = tqdm(self.data_loader, desc=f'Training Epoch {epoch}', 
                   leave=False, ncols=100, unit='batch', 
                   disable=False, dynamic_ncols=True)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)


            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out_x, softmaxed = self.model(data)
                pred = torch.argmax(softmaxed, dim=1)
                loss_target, labelled_mask = self._prepare_target(
                    target, softmaxed.shape[1]
                )
                metric_target, metric_labelled_mask = self._prepare_metric_target(
                    target, softmaxed.shape[1]
                )
                supervised_count = self._supervised_pixel_count(
                    loss_target, labelled_mask
                )
                loss = self.criterion(
                    self._loss_input(self.model, out_x, softmaxed),
                    loss_target,
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}, batch {batch_idx + 1}"
                )

            labelled_count = int(metric_labelled_mask.sum().item())
            labelled_pixels += labelled_count
            ignored_pixels += metric_labelled_mask.numel() - labelled_count
            global_metrics.update(softmaxed, metric_target)

            if supervised_count == 0:
                all_ignored_batches += 1
            elif self.scaler is not None:
                scale_before = self.scaler.get_scale()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), 0.05)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() < scale_before:
                    skipped_optimizer_steps += 1
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.05, error_if_nonfinite=True)
                self.optimizer.step()

            if supervised_count > 0:
                metric_weight = supervised_count if self.mask_unknown_labels else 1
                self.train_metrics.update('loss', loss.item(), n=metric_weight)

            # Update progress bar with current loss
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

            # Disable the original progress logging since we have tqdm
            # if batch_idx % self.log_step == 0:
            #     self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            #         epoch,
            #         self._progress(batch_idx),
            #         loss.item()))

            if batch_idx == self.len_epoch:
                break
        
        pbar.close()
        log = self.train_metrics.result()
        log.update(global_metrics.legacy_names())
        global_result = global_metrics.result()
        log.update(
            {
                key: value
                for key, value in global_result.items()
                if not key.startswith('boundary_')
                and key not in {'accuracy', 'labelled_pixels'}
            }
        )
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            log['max_cuda_memory_gib'] = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            )
        log['epoch_seconds'] = time.perf_counter() - epoch_started
        log['labelled_pixels'] = labelled_pixels
        log['ignored_pixels'] = ignored_pixels
        log['all_ignored_batches'] = all_ignored_batches
        if self.scaler is not None:
            log['amp_skipped_steps'] = skipped_optimizer_steps
            log['grad_scaler_scale'] = float(self.scaler.get_scale())

        if self.do_validation:
            val_log = self._valid_epoch(self.valid_data_loader)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        # Log metrics to TensorBoard
        self._log_to_tensorboard(log, epoch)

        # Write epoch summary to persistent log file
        self._write_to_log(f"\n{'='*60}")
        self._write_to_log(f"Epoch {epoch} Summary")
        self._write_to_log(f"{'='*60}")
        for key, value in log.items():
            self._write_to_log(f"    {key:15s}: {value}")

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _final_evaluation(self, epoch):
        """Test a separate best-model copy after last/best checkpoints commit.

        Never resume here: that would replace the final optimizer, scheduler,
        RNG and model state with the state from an earlier best epoch.
        """
        if not self.do_test:
            return
        best_path = self.checkpoint_dir / 'model_best.pth'
        from utils.training_protocol import load_checkpoint
        checkpoint = load_checkpoint(best_path)
        training_model = self.model
        rng = self._capture_rng_state()
        try:
            self.model = copy.deepcopy(training_model)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            self._paper_artifacts = None
            if self.config.get('visualizations', {}).get('enabled', True):
                from visualisation.paper_artifacts import PaperArtifacts
                self._paper_artifacts = PaperArtifacts(
                    self.model, self.test_data_loader, self.config, best_path)
            self.logger.info("Testing model_best.pth from epoch %s", checkpoint['epoch'])
            log = {'test_' + k: v for k, v in self._valid_epoch(self.test_data_loader).items()}
            self._log_to_tensorboard(log, epoch)
            self._write_to_log(f"\nFinal evaluation: model_best.pth (epoch {checkpoint['epoch']})")
            for key, value in log.items():
                self._write_to_log(f"    {key:15s}: {value}")
            from utils import write_json
            write_json({'checkpoint': 'model_best.pth', 'best_epoch': checkpoint['epoch'],
                        'last_epoch': epoch, 'loss_input': 'logits', 'metrics': log},
                       self.checkpoint_dir / 'final_test_metrics.json')
            self.tb_writer.flush()
            if self._paper_artifacts is not None:
                output = self._paper_artifacts.finish(log)
                self._write_to_log(f'Paper visualizations: {output}')
        finally:
            self._paper_artifacts = None
            self.model = training_model
            self._restore_rng_state(rng)

    def _valid_epoch(self, data_loader):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        labelled_pixels = 0
        ignored_pixels = 0
        all_ignored_batches = 0
        global_metrics = MaskedSegmentationMetrics(
            ignore_index=self.ignore_index,
            boundary_dilation_ratio=self.boundary_dilation_ratio,
            compute_boundary=True,
        )
        
        # Create progress bar for validation
        pbar = tqdm(data_loader, desc='Validation', leave=False, ncols=100, unit='batch', 
                   dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)


                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    out_x, softmaxed = self.model(data)
                    # Evaluation loss and every reported metric use the same
                    # explicitly labelled population, regardless of historical
                    # training-loss compatibility settings.
                    loss_target, labelled_mask = self._prepare_metric_target(
                        target, softmaxed.shape[1]
                    )
                    supervised_count = int(labelled_mask.sum().item())
                    loss = self.criterion(
                        self._loss_input(self.model, out_x, softmaxed),
                        loss_target,
                    )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite validation loss at batch {batch_idx + 1}"
                    )

                labelled_count = int(labelled_mask.sum().item())
                labelled_pixels += labelled_count
                ignored_pixels += labelled_mask.numel() - labelled_count
                if supervised_count == 0:
                    all_ignored_batches += 1
                else:
                    self.valid_metrics.update(
                        'loss', loss.item(), n=supervised_count
                    )
                global_metrics.update(softmaxed, loss_target)
                artifacts = getattr(self, '_paper_artifacts', None)
                if artifacts is not None:
                    artifacts.consume(data, target, softmaxed, evaluation_metrics=global_metrics)

                # Update progress bar with current loss
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})


        pbar.close()
        log = self.valid_metrics.result()
        log.update(global_metrics.legacy_names())
        log.update(global_metrics.result())
        log['labelled_pixels'] = labelled_pixels
        log['ignored_pixels'] = ignored_pixels
        log['all_ignored_batches'] = all_ignored_batches
        log['classification_report'] = global_metrics.classification_report()
        return log

    def _log_to_tensorboard(self, log, epoch):
        """
        Log metrics to TensorBoard
        
        :param log: Dictionary containing metrics to log
        :param epoch: Current epoch number
        """
        for key, value in log.items():
            phase, metric = ('val', key[4:]) if key.startswith('val_') else (
                ('test', key[5:]) if key.startswith('test_') else ('train', key))
            tag = f'{phase}/{metric}'
            if isinstance(value, (int, float)) and key != 'epoch':
                self.tb_writer.add_scalar(tag, value, epoch)
            elif metric == 'classification_report':
                self.tb_writer.add_text(tag, f"Epoch {epoch}\n```\n{value}\n```", epoch)
        
        # Log learning rate if available
        if self.lr_scheduler is not None:
            self.tb_writer.add_scalar('train/learning_rate',
                                    self.lr_scheduler.get_last_lr()[0], epoch)
    
    def close_tensorboard(self):
        """
        Close TensorBoard writer and log file
        """
        self.tb_writer.close()
        self._close_log_file()

    def _init_log_file(self):
        """
        Initialize a persistent training log file in the model save directory.
        """
        self.log_file_path = os.path.join(str(self.checkpoint_dir), 'training_log.txt')
        # Resume reuses the same run directory, so append an explicit boundary
        # to the existing log rather than creating a disconnected log.
        with open(self.log_file_path, 'a') as f:
            f.write(f"\n{'#'*60}\n")
            if self.config.resume is not None:
                f.write(f"# Training Resumed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Checkpoint: {self.config.resume}\n")
                f.write(f"# Resume epoch: {self.start_epoch}\n")
            else:
                f.write(f"# Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'#'*60}\n")

    def _write_to_log(self, text):
        """
        Append a line of text to the persistent training log file.

        :param text: String to write to the log file
        """
        with open(self.log_file_path, 'a') as f:
            f.write(text + '\n')

    def _close_log_file(self):
        """
        Write a closing marker to the log file.
        """
        with open(self.log_file_path, 'a') as f:
            f.write(f"\n{'#'*60}\n")
            f.write(f"# Training Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'#'*60}\n")

    def _save_checkpoint(self, epoch, save_best=False, is_last=False):
        """
        Saving checkpoints with lr_scheduler state

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        :param is_last: if True, save as 'model_last_epoch{}.pth' and clean up old last checkpoints
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'rng_state': self._capture_rng_state(),
            'rng_state_version': 1,
            'precision': self.precision.mode,
        }

        if self.scaler is not None:
            state['grad_scaler'] = self.scaler.state_dict()
        
        # Add lr_scheduler state if available
        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        if is_last:
            # Atomically commit the new checkpoint before removing the old one.
            import glob
            filename = str(self.checkpoint_dir / 'model_last_epoch{}.pth'.format(epoch))
            self._atomic_torch_save(state, filename)
            self._atomic_torch_save(state, self.checkpoint_dir / 'model_last.pth')
            old_last_files = glob.glob(str(self.checkpoint_dir / 'model_last_epoch*.pth'))
            for old_file in old_last_files:
                if os.path.abspath(old_file) == os.path.abspath(filename):
                    continue
                try:
                    os.remove(old_file)
                except OSError:
                    pass
            self.logger.info("Saving last checkpoint: {} ...".format(filename))
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            self._atomic_torch_save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
            
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            self._atomic_torch_save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints with lr_scheduler state

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        from utils.training_protocol import load_checkpoint
        checkpoint = load_checkpoint(resume_path)
        if 'epoch' not in checkpoint:
            self.model.load_state_dict(checkpoint, strict=False)

        if 'epoch' in checkpoint:
            if self.config.get('protocol_id') is not None:
                from utils.training_protocol import assert_resume_compatible
                assert_resume_compatible(self.config, checkpoint['config'])
            checkpoint_precision = checkpoint.get('precision')
            if checkpoint_precision is not None and checkpoint_precision != self.precision.mode:
                raise ValueError(
                    "Cannot resume a checkpoint with a different numeric precision: "
                    f"checkpoint={checkpoint_precision}, requested={self.precision.mode}"
                )
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']
            # load architecture params from checkpoint.
            if checkpoint['config']['arch'] != self.config['arch']:
                self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                    "checkpoint. This may yield an exception while state_dict is being loaded.")
            self.model.load_state_dict(checkpoint['state_dict'])

            # load optimizer state from checkpoint only when optimizer type is not changed.
            if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
                self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                    "Optimizer parameters not being resumed.")
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            # load lr_scheduler state from checkpoint if available
            if 'lr_scheduler' in checkpoint and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.logger.info("Learning rate scheduler state loaded from checkpoint.")

            if self.scaler is not None:
                if 'grad_scaler' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['grad_scaler'])
                    self.logger.info("FP16 gradient scaler state loaded from checkpoint.")
                else:
                    self.logger.warning(
                        "FP16 checkpoint has no gradient scaler state; resume may "
                        "not match an uninterrupted run."
                    )

            self._restore_rng_state(checkpoint.get('rng_state'))

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
