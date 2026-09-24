# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import hashlib
import pathlib
import random
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import WarmupPolynomialLR, prepare_device, resolve_precision


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

# Experiment protocol uses strict FP32 and deterministic cuDNN rather than
# AMP, TF32, or autotuned kernels whose selection can vary between runs.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main(config):
    from utils.training_protocol import validate_training_config
    validate_training_config(config)
    logger = config.get_logger('train')

    # fix random seeds dynamically
    seed = config.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Using random seed: {seed}")


    # setup data_loader instances
    train_data_loader = config.init_obj('train_data_loader', module_data)
    val_data_loader = config.init_obj('train_data_loader', module_data, mode='val')
    test_data_loader = config.init_obj('train_data_loader', module_data, mode='test')

    # print data shapes
    print(f"Train data shape: {train_data_loader.dataset[0][0].shape}")
    print(f"Val data shape: {val_data_loader.dataset[0][0].shape}")
    print(f"Test data shape: {test_data_loader.dataset[0][0].shape}")

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    initial_state_path = config['trainer'].get('initial_state_path')
    initial_state_sha256 = config['trainer'].get('initial_state_sha256')
    if initial_state_path is not None and config.resume is None:
        if config['trainer'].get('pretrained_model') is not None:
            raise ValueError(
                "trainer.initial_state_path cannot be combined with "
                "trainer.pretrained_model"
            )
        initial_state_path = pathlib.Path(initial_state_path)
        if not initial_state_path.is_file():
            raise FileNotFoundError(f"Initial model state not found: {initial_state_path}")
        original_posix = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        try:
            initial_payload = torch.load(
                initial_state_path, map_location='cpu', weights_only=False
            )
        finally:
            pathlib.PosixPath = original_posix
        initial_state = initial_payload.get('state_dict', initial_payload)
        model.load_state_dict(initial_state, strict=True)
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update(name.encode('utf-8'))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        initial_state_sha256 = digest.hexdigest()
        expected_hash = config['trainer'].get('initial_state_sha256')
        if expected_hash is not None and expected_hash.lower() != initial_state_sha256.lower():
            raise ValueError(
                "Initial model state hash mismatch: "
                f"expected={expected_hash}, actual={initial_state_sha256}"
            )
        logger.info(
            f"Loaded canonical initial model state: {initial_state_path} "
            f"(state_sha256={initial_state_sha256})"
        )
        del initial_payload, initial_state
    elif initial_state_path is not None:
        logger.info(
            "Resume requested; canonical initial state is retained as provenance "
            "but the checkpoint state takes precedence."
        )
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    precision = resolve_precision(config['trainer'], device)
    precision_mode = precision.display_name
    logger.info(
        f"Numeric precision: {precision_mode}; "
        f"autocast={precision.use_autocast}; "
        f"GradScaler={precision.use_grad_scaler}; "
        f"TF32 matmul={torch.backends.cuda.matmul.allow_tf32 if device.type == 'cuda' else False}; "
        f"TF32 cuDNN={torch.backends.cudnn.allow_tf32 if device.type == 'cuda' else False}"
    )
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Optional PyTorch 2.x torch.compile optimization
    if config['trainer'].get('compile', False) and hasattr(torch, 'compile'):
        logger.info("Compiling model with PyTorch 2.x torch.compile() ...")
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    logger.info("Using standard (non-fused) optimizer for reproducibility.")

    scheduler_config = config['lr_scheduler']
    if scheduler_config['type'] == 'WarmupPolynomialLR':
        total_steps = config['trainer']['epochs'] * len(train_data_loader)
        lr_scheduler = WarmupPolynomialLR(
            optimizer,
            total_steps=total_steps,
            **scheduler_config['args'],
        )
        logger.info(
            "Using optimizer-step LR schedule: total_steps=%s, "
            "warmup_steps=%s, power=%s",
            lr_scheduler.total_steps,
            lr_scheduler.warmup_steps,
            lr_scheduler.power,
        )
    else:
        lr_scheduler = config.init_obj(
            'lr_scheduler', torch.optim.lr_scheduler, optimizer
        )


    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=val_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler)

    # Write data shapes to the persistent log file
    # Dataset indexing performs random augmentation in train mode. Preserve the
    # NumPy RNG state so diagnostic logging cannot change the training stream.
    numpy_rng_state = np.random.get_state()
    try:
        trainer._write_to_log(f"Train data shape: {train_data_loader.dataset[0][0].shape}")
        trainer._write_to_log(f"Val data shape: {val_data_loader.dataset[0][0].shape}")
        trainer._write_to_log(f"Test data shape: {test_data_loader.dataset[0][0].shape}")
    finally:
        np.random.set_state(numpy_rng_state)
    trainer._write_to_log(f"Model architecture: {type(model).__name__}")
    trainer._write_to_log(f"Optimizer: {config['optimizer']['type']}")
    trainer._write_to_log(f"Learning rate: {config['optimizer']['args']['lr']}")
    trainer._write_to_log(f"Gradient clipping max norm: {trainer.grad_clip_max_norm}")
    trainer._write_to_log(
        f"LR scheduler interval: {config['lr_scheduler'].get('interval', 'epoch')}"
    )
    if isinstance(lr_scheduler, WarmupPolynomialLR):
        trainer._write_to_log(f"LR total optimizer steps: {lr_scheduler.total_steps}")
        trainer._write_to_log(f"LR warmup steps: {lr_scheduler.warmup_steps}")
        trainer._write_to_log(f"LR polynomial power: {lr_scheduler.power}")
    trainer._write_to_log(f"Epochs: {config['trainer']['epochs']}")
    trainer._write_to_log(f"Seed: {seed}")
    trainer._write_to_log(f"Initial state path: {initial_state_path}")
    trainer._write_to_log(f"Initial state SHA-256: {initial_state_sha256}")
    trainer._write_to_log(f"Numeric precision: {precision_mode}")
    trainer._write_to_log(f"Autocast enabled: {precision.use_autocast}")
    trainer._write_to_log(f"Autocast dtype: {precision.autocast_dtype}")
    trainer._write_to_log(f"GradScaler enabled: {precision.use_grad_scaler}")
    trainer._write_to_log(
        f"Unknown-label masking enabled: {trainer.mask_unknown_labels}"
    )
    trainer._write_to_log(f"Unknown source label: {trainer.unknown_label}")
    trainer._write_to_log(f"Loss/metric ignore index: {trainer.ignore_index}")
    trainer._write_to_log(
        f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32 if device.type == 'cuda' else False}"
    )
    trainer._write_to_log(
        f"TF32 cuDNN: {torch.backends.cudnn.allow_tf32 if device.type == 'cuda' else False}"
    )

    if config['trainer']['mode'] == 'train':
        trainer.train()
    else:
        log = trainer._valid_epoch(test_data_loader)
        for key, value in log.items():
            logger.info('    test_{:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='U-Net Forest Segmentation Trainer')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--run_id', default=None, type=str,
                      help='explicit unique run identifier (default: timestamp)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='train_data_loader;args;batch_size'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--topology'], type=str, target='arch;args;topology'),
        CustomArgs(['--save_dir'], type=str, target='trainer;save_dir'),
        CustomArgs(['--seed'], type=int, target='seed'),
        CustomArgs(['--num_workers', '--nw'], type=int, target='train_data_loader;args;num_workers'),
        CustomArgs(['--precision'], type=str, target='trainer;precision'),
    ]

    config = ConfigParser.from_args(args, options, require_training_protocol=True)
    main(config)
