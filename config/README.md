# Training configurations

Only the current corrected protocol is kept in this repository:

- `protocol_v3/*_MS.json`: 18-band multispectral runs.
- `protocol_v3/*_RGB.json`: Landsat bands 4/3/2 runs.
- `protocol_v3/bigdata_UNet_MS_seed42.json`: current large-dataset UNet setup.
- `protocol_v3/UNetMFFSE_MS_normalized_masked_dropout06_seed42.json`: explicit
  dropout-0.6 comparison using the otherwise current protocol.
- `protocol_v3/artifacts/`: frozen split manifests and valid-training-pixel
  normalization statistics required by the configs.

The 18 paired model configs deliberately map one-to-one to the nine architecture
factories and two modalities. Seeds, epochs, batch size, learning rate and save
location can be overridden on the command line instead of creating another
near-duplicate config, for example:

```powershell
.venv\Scripts\python.exe train.py `
  --config config/protocol_v3/UNet_MS.json `
  --epochs 60 --seed 123 --run_id unet_ms_seed123
```

TensorBoard is always enabled by the trainer and writes into each run's
`tensorboard_logs/` directory. The obsolete `trainer.tensorboard: false` field
was removed because it never controlled the writer.

Protocol-v3 training monitors validation loss and stops after 10 consecutive
epochs without improvement. The non-improvement counter is checkpointed so a
resumed run preserves the same patience window as uninterrupted training.
All protocol-v3 runs use gradient clipping with `max_norm=0.5`; the value is
stored in each config and checked when resuming so it cannot drift silently.

Custom SegFormer uses a pretrained MiT encoder and randomly initialized
decoder, AdamW (`lr=6e-5`, `betas=(0.9, 0.999)`, `weight_decay=0.01`), and a
step-based linear-warmup/polynomial-decay schedule. Warmup is calculated as 5%
of the maximum optimizer steps (`epochs * batches_per_epoch`) rather than a
fixed step count. Its data, augmentation, focal loss, masking, early stopping,
and best-checkpoint rules remain shared with the UNet protocol.

Historical, AMP, normalization-benchmark, reproduction and obsolete root
configs are archived outside the repository and are not runtime dependencies.
