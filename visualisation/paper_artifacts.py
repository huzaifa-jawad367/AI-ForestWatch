"""Run-scoped, bounded-memory paper artifacts from the actual evaluation pass.

No historical checkpoint defaults, no model reconstruction, no label-derived
input normalization. SE diagnostics are descriptive, not a trained ablation.
"""
from pathlib import Path
from datetime import datetime, timezone
import re
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from torchvision.ops import SqueezeExcitation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from PIL import Image

from model.segmentation_metrics import MaskedSegmentationMetrics
from utils import encode_segmentation_target, write_json

LABEL_COLORS = np.array([[30, 41, 59], [229, 62, 62], [56, 161, 105]], dtype=np.uint8)
ERROR_COLORS = np.array([[30, 41, 59], [180, 180, 180], [56, 161, 105],
                         [245, 158, 11], [220, 38, 38]], dtype=np.uint8)


def _figure(rows=1, cols=1, size=None):
    fig = Figure(figsize=size or (4 * cols, 3.5 * rows), layout='constrained')
    FigureCanvasAgg(fig)
    return fig, fig.subplots(rows, cols, squeeze=False)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.' + extension), dpi=250, bbox_inches='tight')
    fig.clear()


def error_codes(prediction, truth):
    """0 unknown, 1 TN, 2 TP, 3 FP, 4 FN; unknown is never an error."""
    result = np.zeros(truth.shape, dtype=np.uint8)
    result[(truth == 1) & (prediction == 0)] = 1
    result[(truth == 2) & (prediction == 1)] = 2
    result[(truth == 1) & (prediction == 1)] = 3
    result[(truth == 2) & (prediction == 0)] = 4
    return result


def artifact_directory(config, checkpoint):
    checkpoint = Path(checkpoint)
    root = Path(config.get('visualizations', {}).get(
        'output_root', str(Path(config['trainer']['save_dir']) / 'error_maps')))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    return root / config['name'] / config['arch']['type'] / checkpoint.parent.name / stamp


class PaperArtifacts:
    def __init__(self, model, loader, config, checkpoint):
        if not isinstance(loader.sampler, SequentialSampler):
            raise ValueError('Paper artifacts require sequential test sampling for traceable file IDs')
        self.model, self.loader, self.config = model, loader, config
        self.checkpoint = Path(checkpoint)
        self.output = artifact_directory(config, checkpoint)
        self.output.mkdir(parents=True, exist_ok=False)
        self.options = config.get('visualizations', {})
        self.limit = int(self.options.get('max_qualitative_samples', 12))
        if self.limit < 1:
            raise ValueError('max_qualitative_samples must be positive')
        self.offset, self.batch = 0, 0
        self.records, self.selected, self.skipped = [], [], []
        self.metrics = MaskedSegmentationMetrics(
            boundary_dilation_ratio=config.get('trainer', {}).get('boundary_dilation_ratio', 0.02))
        self.se_modules = {name: module for name, module in model.named_modules()
                           if isinstance(module, SqueezeExcitation)}
        dataset = loader.dataset
        if hasattr(dataset, 'all_images'):
            first = {}
            for index, (source, row, col) in enumerate(dataset.all_images):
                first.setdefault(str(source), index)
            candidates = [first[key] for key in sorted(first)]
        else:
            candidates = list(range(len(dataset)))
        positions = np.linspace(0, len(candidates) - 1, min(self.limit, len(candidates)), dtype=int)
        self.selected_indices = {candidates[i] for i in positions}
        write_json({'status': 'in_progress', 'checkpoint': str(checkpoint)}, self.output / 'manifest.json')

    def _identity(self, index):
        dataset = self.loader.dataset
        if hasattr(dataset, 'all_images'):
            source, row, col = dataset.all_images[index]
            return {'index': index, 'source_id': Path(source).name, 'row': int(row), 'col': int(col)}
        return {'index': index, 'source_id': f'sample_{index}', 'row': 0, 'col': 0}

    def _rgb(self, index, data):
        dataset = self.loader.dataset
        if hasattr(dataset, 'all_images') and hasattr(dataset, 'cache'):
            source, row, col = dataset.all_images[index]
            raw = dataset.cache[source][0]
            size = dataset.model_input_size
            rgb = raw[row:row + size, col:col + size, [3, 2, 1]].astype(np.float32)
        else:
            bands = self.config.get('train_data_loader', {}).get('args', {}).get('bands', [])
            if not all(band in bands for band in (4, 3, 2)):
                return None
            rgb = data[[bands.index(band) for band in (4, 3, 2)]].numpy().transpose(1, 2, 0)
            normalization = getattr(dataset, 'normalization', None)
            if normalization is not None:
                mean, std = normalization
                indices = [bands.index(band) for band in (4, 3, 2)]
                rgb = rgb * std[indices] + mean[indices]
        rgb = np.nan_to_num(rgb)
        # Display stretch only, never fed back into the model.
        low, high = np.percentile(rgb, [2, 98], axis=(0, 1))
        return np.clip((rgb - low) / np.maximum(high - low, 1e-8), 0, 1)

    @torch.no_grad()
    def consume(self, data, target, probabilities, evaluation_metrics=None):
        probabilities = probabilities.detach().float().cpu()
        data, target = data.detach(), target.detach().cpu()
        encoded, _ = encode_segmentation_target(target, mask_unknown_labels=True)
        if evaluation_metrics is None:
            self.metrics.update(probabilities, encoded)
        else:
            # Reuse the evaluation's exact counters instead of recomputing
            # expensive boundary metrics on CPU for every crop.
            self.metrics = evaluation_metrics
        truth = np.where(encoded.numpy() == -100, 0, encoded.numpy() + 1).astype(np.uint8)
        prediction = probabilities.argmax(1).numpy().astype(np.uint8)
        forest = probabilities[:, 1].numpy()
        errors = np.stack([error_codes(p, t) for p, t in zip(prediction, truth)])
        destination = self.output / 'predictions'
        destination.mkdir(exist_ok=True)
        np.savez_compressed(destination / f'batch_{self.batch:05d}.npz',
                            prediction=prediction, forest_probability=forest,
                            truth=truth, errors=errors,
                            dataset_indices=np.arange(self.offset, self.offset + len(truth)))
        for local in range(len(truth)):
            index = self.offset + local
            identity = self._identity(index)
            identity.update(batch_file=f'predictions/batch_{self.batch:05d}.npz', batch_offset=local)
            self.records.append(identity)
            if index in self.selected_indices:
                selected_data = data[local].cpu().clone()
                self.selected.append(dict(identity=identity, data=selected_data,
                    rgb=self._rgb(index, selected_data), truth=truth[local], prediction=prediction[local],
                    probability=forest[local], errors=errors[local]))
        self.offset += len(truth)
        self.batch += 1

    def _qualitative(self):
        fig, axes = _figure(len(self.selected), 5, (15, 2.8 * len(self.selected)))
        for row, sample in enumerate(self.selected):
            identity = sample['identity']
            name = f"{identity['index']:06d}_{Path(identity['source_id']).stem}_r{identity['row']}_c{identity['col']}"
            folder = self.output / 'qualitative' / name
            folder.mkdir(parents=True)
            valid = sample['truth'] > 0
            predicted = np.where(valid, sample['prediction'] + 1, 0)
            images = {'truth': LABEL_COLORS[sample['truth']], 'prediction': LABEL_COLORS[predicted],
                      'errors': ERROR_COLORS[sample['errors']]}
            if sample['rgb'] is not None:
                images['rgb'] = (sample['rgb'] * 255).astype(np.uint8)
            for key, pixels in images.items():
                Image.fromarray(pixels).save(folder / f'{key}.png')
            p = np.clip(sample['probability'], 1e-7, 1 - 1e-7)
            entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            detail, panels = _figure(1, 2)
            for ax, image, title in zip(panels.flat, (p, entropy), ('Forest probability', 'Predictive entropy (bits)')):
                handle = ax.imshow(image, vmin=0, vmax=1, cmap='viridis')
                ax.set_title(title); ax.axis('off'); detail.colorbar(handle, ax=ax)
            _save(detail, folder / 'probability_uncertainty')
            panels_data = [images.get('rgb'), images['truth'], images['prediction'], p, images['errors']]
            for ax, image, title in zip(axes[row], panels_data, ('RGB (display stretch)', 'Ground truth', 'Prediction (valid area)', 'Forest probability', 'Error map')):
                if image is not None:
                    ax.imshow(image, cmap='viridis', vmin=0, vmax=1) if image.ndim == 2 else ax.imshow(image)
                else:
                    ax.text(.5, .5, 'RGB unavailable', ha='center')
                ax.set_title(title); ax.axis('off')
            axes[row, 0].set_ylabel(name)
            axes[row, 0].set_title(f'RGB (display stretch)\n{identity["source_id"]} [{identity["row"]}, {identity["col"]}]', fontsize=8)
        fig.suptitle('Test examples: deterministic file-based selection\nErrors: grey TN | green TP | orange FP | red FN | slate unknown (excluded)', fontsize=11)
        _save(fig, self.output / 'qualitative_comparison')

    def _summary_plots(self):
        matrix = self.metrics.confusion.numpy()
        fig, axes = _figure(1, 2)
        normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
        for ax, values, title in zip(axes.flat, (matrix, normalized), ('Valid-pixel confusion counts', 'Row-normalized confusion')):
            ax.imshow(values, cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(values[i, j]) if values.dtype.kind in 'iu' else f'{values[i, j]:.4f}', ha='center')
            ax.set_xticks([0, 1], ['Non-Forest', 'Forest']); ax.set_yticks([0, 1], ['Non-Forest', 'Forest'])
            ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
        _save(fig, self.output / 'confusion_matrix')
        result = self.metrics.result()
        fig, axes = _figure(1, 3, (12, 4))
        for ax, metric in zip(axes.flat, ('iou', 'dice', 'boundary_iou')):
            values = [result[f'{metric}_non_forest'], result[f'{metric}_forest'], result[f'mean_{metric}']]
            ax.bar(['Non-Forest', 'Forest', 'Mean'], values)
            ax.set_ylim(0, 1); ax.set_title(metric.replace('_', ' ').title())
        _save(fig, self.output / 'class_metrics')
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_dir = self.checkpoint.parent / 'tensorboard_logs'
        if not event_dir.exists():
            self.skipped.append('Learning curves: TensorBoard event directory unavailable')
            return
        events = EventAccumulator(str(event_dir), size_guidance={'scalars': 0}).Reload()
        tags = events.Tags()['scalars']
        fig, axes = _figure(1, 3, (13, 4))
        count = 0
        for ax, metric in zip(axes.flat, ('loss', 'mean_iou', 'mean_dice')):
            for phase in ('train', 'val'):
                new_tag, old_tag = f'{phase}/{metric}', metric if phase == 'train' else f'val_{metric}'
                history = {}
                for tag in (old_tag, new_tag):
                    if tag in tags:
                        history.update({event.step: event.value for event in events.Scalars(tag)})
                if history:
                    steps = sorted(history)
                    ax.plot(steps, [history[step] for step in steps], label=phase); count += 1
            ax.set_title(metric); ax.set_xlabel('Epoch')
            if ax.lines: ax.legend()
        if count:
            _save(fig, self.output / 'learning_curves')
        else:
            fig.clear(); self.skipped.append('Learning curves: no supported scalar tags')

    @torch.no_grad()
    def _se_plots(self):
        if not self.se_modules:
            self.skipped.append('SE diagnostics: architecture has no SqueezeExcitation modules')
            return
        device = next(self.model.parameters()).device
        captures, handles = {}, []
        all_features = {name: [] for name in self.se_modules}
        for name, module in self.se_modules.items():
            def capture(mod, inputs, output, name=name):
                captures[name] = (F.adaptive_avg_pool2d(inputs[0].detach().float(), (8, 8)).cpu(),
                                  F.adaptive_avg_pool2d(output.detach().float(), (8, 8)).cpu())
            handles.append(module.register_forward_hook(capture))
        try:
            for sample in self.selected:
                captures.clear()
                _, on_probabilities = self.model(sample['data'].unsqueeze(0).to(device))
                sample['se_on_prediction'] = on_probabilities.argmax(1)[0].cpu().numpy()
                for name, (before, after) in captures.items():
                    labels = F.interpolate(torch.as_tensor(sample['truth']).view(1, 1, *sample['truth'].shape).float(),
                                           size=(8, 8), mode='nearest').flatten().numpy()
                    all_features[name].append((before[0].numpy(), after[0].numpy(), labels))
        finally:
            for handle in handles: handle.remove()
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        for name, features in all_features.items():
            if not features:
                self.skipped.append(f'SE module not invoked: {name}'); continue
            folder = self.output / 'se' / re.sub(r'[^a-zA-Z0-9_-]', '_', name)
            folder.mkdir(parents=True)
            before = np.concatenate([item[0].reshape(item[0].shape[0], -1).T for item in features])
            after = np.concatenate([item[1].reshape(item[1].shape[0], -1).T for item in features])
            labels = np.concatenate([item[2] for item in features])
            valid = labels > 0
            before, after, labels = before[valid], after[valid], labels[valid]
            np.savez_compressed(folder / 'sampled_features.npz', before=before, after=after, labels=labels)
            first_b, first_a, _ = features[0]
            fig, axes = _figure(1, 3)
            lo, hi = min(first_b.mean(0).min(), first_a.mean(0).min()), max(first_b.mean(0).max(), first_a.mean(0).max())
            for ax, image, title in zip(axes.flat, (first_b.mean(0), first_a.mean(0), (first_a-first_b).mean(0)), ('Before SE', 'After SE', 'After minus before')):
                handle = ax.imshow(image, cmap='viridis', vmin=lo if title != 'After minus before' else None,
                                   vmax=hi if title != 'After minus before' else None)
                ax.set_title(title); fig.colorbar(handle, ax=ax)
            _save(fig, folder / 'feature_maps')
            if len(before) < 4 or before.shape[1] < 2:
                self.skipped.append(f'{name}: insufficient labelled features for correlation/embedding'); continue
            fig, axes = _figure(1, 2)
            for ax, values, title in zip(axes.flat, (before, after), ('Before SE', 'After SE')):
                centered = values - values.mean(0)
                scale = np.linalg.norm(centered, axis=0)
                standardized = centered / np.maximum(scale, 1e-12)
                correlation = standardized.T @ standardized
                handle = ax.imshow(correlation, vmin=-1, vmax=1, cmap='coolwarm')
                ax.set_title(title); fig.colorbar(handle, ax=ax)
            _save(fig, folder / 'channel_correlation')
            if not np.any(np.std(np.concatenate((before, after)), axis=0) > 1e-10):
                self.skipped.append(f'{name}: constant features, embedding/gate plots not meaningful')
                continue
            # Shared PCA axes; independent t-SNE layouts are descriptive only.
            pca = PCA(2, random_state=42).fit(np.concatenate((before, after)))
            common_pca = pca.transform(np.concatenate((before, after)))
            fig, axes = _figure(2, 2)
            for column, values in enumerate((before, after)):
                embedded = (pca.transform(values), TSNE(n_components=2, perplexity=min(30, len(values)-1),
                            random_state=42, init='random', learning_rate='auto').fit_transform(values))
                for row, points in enumerate(embedded):
                    ax = axes[row, column]
                    for label, color in ((1, '#E53E3E'), (2, '#38A169')):
                        selected = labels == label
                        ax.scatter(points[selected, 0], points[selected, 1], s=3, color=color,
                                   label='Non-Forest' if label == 1 else 'Forest', rasterized=True)
                    ax.set_title(('PCA' if row == 0 else 't-SNE') + (' before SE' if column == 0 else ' after SE'))
                    if row == 0:
                        low, high = common_pca.min(0), common_pca.max(0)
                        pad = np.maximum((high-low) * .05, 1e-6)
                        ax.set_xlim(low[0]-pad[0], high[0]+pad[0])
                        ax.set_ylim(low[1]-pad[1], high[1]+pad[1])
                    ax.legend()
            _save(fig, folder / 'feature_embeddings')
            # Ratio is gate strength where pre-SE activations are nonzero.
            gate = np.divide(after, before, out=np.full_like(after, np.nan), where=np.abs(before) > 1e-8)
            valid_gate = np.isfinite(gate)
            counts = valid_gate.sum(0)
            means = np.nansum(gate, axis=0) / np.maximum(counts, 1)
            means[counts == 0] = np.nan
            fig, axes = _figure()
            axes[0, 0].plot(means); axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_xlabel('Latent channel (not input band)'); axes[0, 0].set_ylabel('Mean observed SE gate')
            axes[0, 0].set_title('Channel attenuation / retention on selected examples')
            _save(fig, folder / 'channel_gates')
            np.savez_compressed(folder / 'channel_gates.npz', mean=means, observed_pixels=counts)
            finite = np.flatnonzero(np.isfinite(means))
            if len(finite):
                order = finite[np.argsort(means[finite])]
                channels = np.concatenate((order[:min(3, len(order))], order[-min(3, len(order)):]))
                fig, axes = _figure(2, len(channels)//2)
                for ax, channel in zip(axes.flat, channels):
                    ax.imshow(first_b[channel], cmap='viridis')
                    ax.set_title(f'Channel {channel}: mean gate {means[channel]:.3f}'); ax.axis('off')
                fig.suptitle('Lower-retention channels (top), higher-retention channels (bottom)')
                _save(fig, folder / 'channel_roles')
        # Disable all SE modules by replacing their outputs with their inputs.
        # Hooks are removed even on failure; this is NOT a separately trained UNet.
        handles = [module.register_forward_hook(lambda mod, inputs, output: inputs[0])
                   for module in self.se_modules.values()]
        try:
            fig, axes = _figure(len(self.selected), 2, (8, 3 * len(self.selected)))
            scene_deltas, coverages = [], []
            for row, sample in enumerate(self.selected):
                _, probabilities = self.model(sample['data'].unsqueeze(0).to(device))
                off = probabilities.argmax(1)[0].cpu().numpy()
                # On and bypass passes use the same FP32 path, even if the
                # primary evaluation explicitly requested autocast.
                valid = sample['truth'] > 0
                delta = np.zeros(off.shape, dtype=np.int8)
                delta[valid] = ((sample['se_on_prediction'][valid] == sample['truth'][valid]-1).astype(np.int8)
                                - (off[valid] == sample['truth'][valid]-1).astype(np.int8))
                scene_deltas.append(float(delta[valid].mean()) if valid.any() else float('nan'))
                coverages.append(float((sample['truth'][valid] == 2).mean()) if valid.any() else float('nan'))
                np.savez_compressed(self.output / 'se' / f"bypass_{sample['identity']['index']:06d}.npz",
                                    prediction_on=sample['se_on_prediction'], prediction_bypass=off,
                                    valid=valid, correctness_delta=delta)
                axes[row, 0].imshow(LABEL_COLORS[np.where(valid, off+1, 0)])
                axes[row, 1].imshow(np.ma.masked_where(~valid, delta), vmin=-1, vmax=1,
                                   cmap=ListedColormap(['#ef4444', '#e5e7eb', '#22c55e']))
                axes[row, 0].set_title('SE bypassed at inference'); axes[row, 1].set_title('SE on minus bypass: red hurt, green helped')
            _save(fig, self.output / 'se' / 'inference_bypass_diagnostic')
            fig, axes = _figure()
            axes[0, 0].scatter(np.asarray(coverages)*100, np.asarray(scene_deltas)*100)
            axes[0, 0].axhline(0, color='grey', linestyle='--')
            axes[0, 0].set_xlabel('Forest coverage on labelled pixels (%)')
            axes[0, 0].set_ylabel('SE on minus bypass accuracy (percentage points)')
            axes[0, 0].set_title('Selected-example diagnostic only; not a trained ablation')
            _save(fig, self.output / 'se' / 'scene_analysis')
        finally:
            for handle in handles: handle.remove()

    def finish(self, metrics=None):
        if self.offset != len(self.loader.dataset):
            raise ValueError('Incomplete evaluation: refusing a complete artifact manifest')
        if not self.selected:
            raise ValueError('No evaluation examples available for paper figures')
        self._qualitative()
        self._summary_plots()
        self._se_plots()
        write_json(self.records, self.output / 'prediction_index.json')
        if metrics is not None:
            write_json(metrics, self.output / 'evaluation_metrics.json')
        self.skipped.append('Cross-model comparison / trained SE ablation / aggregate seed plots: require explicitly matched runs')
        manifest = {'status': 'complete', 'checkpoint': str(self.checkpoint),
                    'protocol_id': self.config.get('protocol_id'), 'seed': self.config.get('seed'),
                    'samples': self.offset, 'selected_examples': [s['identity'] for s in self.selected],
                    'selection': 'first crop per source, evenly spaced over sorted source IDs; not ranked by accuracy',
                    'se_scope': 'selected examples; 8x8 pooled features; diagnostic bypass is not a trained ablation',
                    'split_manifest_sha256': self.config.get('train_data_loader', {}).get('args', {}).get('split_manifest_sha256'),
                    'normalization_sha256': self.config.get('train_data_loader', {}).get('args', {}).get('normalization_sha256'),
                    'raw_encoding': {'prediction': '0 non-forest, 1 forest (unmasked)',
                                     'truth': '0 unknown, 1 non-forest, 2 forest',
                                     'errors': '0 ignored, 1 TN, 2 TP, 3 FP, 4 FN'},
                    'skipped': self.skipped,
                    'files': sorted(str(p.relative_to(self.output)) for p in self.output.rglob('*') if p.is_file())}
        if self.checkpoint.is_file():
            from utils.training_protocol import sha256_file
            manifest['checkpoint_sha256'] = sha256_file(self.checkpoint)
        write_json(manifest, self.output / 'manifest.json')
        return self.output
