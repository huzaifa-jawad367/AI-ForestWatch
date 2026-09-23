# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import tempfile

import numpy as np
import torch
from abc import abstractmethod
from numpy import inf
from pathlib import Path

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.not_improved_count = 0
        self.early_stop = inf

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', 10)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.models_dir

        # checkpoint provided to train.py through command line
        if config.resume is not None:
            self._resume_checkpoint(config.resume)
        # pretrained model specified in config.json
        elif cfg_trainer.get('pretrained_model') is not None:
            self._resume_checkpoint(cfg_trainer['pretrained_model'])

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        last_epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0
                    best = True
                    if hasattr(self, '_write_to_log'):
                        self._write_to_log(f"  >> New best model! {self.mnt_metric}: {self.mnt_best}")
                else:
                    self.not_improved_count += 1

                if self.not_improved_count >= self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    if hasattr(self, '_write_to_log'):
                        self._write_to_log(f"\nEarly stopping triggered after {self.early_stop} epochs without improvement.")
                    self._save_checkpoint(epoch, save_best=best, is_last=True)
                    last_epoch = epoch
                    break

            # Save the most recent checkpoint as model_last.pth (overwrites previous)
            self._save_checkpoint(epoch, save_best=best, is_last=True)
            last_epoch = epoch

        # Checkpoint selection includes the final epoch. Evaluation must not
        # mutate the state that was just saved, nor precede best selection.
        if hasattr(self, '_final_evaluation'):
            self._final_evaluation(last_epoch)
        
        # Close TensorBoard writer if it exists
        if hasattr(self, 'close_tensorboard'):
            self.close_tensorboard()

    def _save_checkpoint(self, epoch, save_best=False, is_last=False):
        """
        Saving checkpoints

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
            'early_stop_not_improved': getattr(self, 'not_improved_count', 0),
        }
        
        if is_last:
            # Commit the new checkpoint before removing the previous one. This
            # leaves at least one complete resumable checkpoint after a crash.
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

    @staticmethod
    def _atomic_torch_save(state, destination):
        """Durably write a checkpoint and atomically replace its destination."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp",
            dir=str(destination.parent),
        )
        try:
            with os.fdopen(fd, 'wb') as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, destination)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _capture_rng_state():
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': None,
        }
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        if not state:
            self.logger.warning(
                "Checkpoint has no RNG state; resume is supported but will not "
                "be bit-for-bit equivalent to uninterrupted training."
            )
            return
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'])
        if state.get('torch_cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['torch_cuda'])
        self.logger.info("Python, NumPy, PyTorch CPU, and CUDA RNG states restored.")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, weights_only=False)
        if not 'epoch' in checkpoint:
            self.model.load_state_dict(torch.load(resume_path, weights_only=False), strict=False)
        else:
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']
            self.not_improved_count = int(
                checkpoint.get('early_stop_not_improved', 0)
            )
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

            self._restore_rng_state(checkpoint.get('rng_state'))

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
