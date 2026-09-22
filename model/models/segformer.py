from transformers import SegformerForSemanticSegmentation, SegformerModel, SegformerDecodeHead
from transformers.models.segformer.configuration_segformer import SegformerConfig
import torch
import torch.nn as nn
from base import BaseModel
from torch.optim import *
from torchvision import models

class CustomSegformer(nn.Module):
    """SegFormer with optional fixed, channel-wise input normalization."""

    loss_uses_logits = True

    def __init__(self, input_channels, num_classes, base_model='nvidia/mit-b3',
                 input_mean=None, input_std=None, input_clip=10.0):
        super().__init__()
        if (input_mean is None) != (input_std is None):
            raise ValueError("input_mean and input_std must be provided together")
        if input_mean is not None:
            if len(input_mean) != input_channels or len(input_std) != input_channels:
                raise ValueError("normalization statistics must match input_channels")
            mean = torch.as_tensor(input_mean, dtype=torch.float32).view(1, input_channels, 1, 1)
            std = torch.as_tensor(input_std, dtype=torch.float32).view(1, input_channels, 1, 1)
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("normalization statistics must be finite and std must be positive")
        else:
            mean = torch.empty(0, dtype=torch.float32)
            std = torch.empty(0, dtype=torch.float32)
        if input_clip is not None and input_clip <= 0:
            raise ValueError("input_clip must be positive or None")
        # Stats live in the config, so keep these buffers out of checkpoints to
        # preserve compatibility with older SegFormer state dictionaries.
        self.register_buffer("input_mean", mean, persistent=False)
        self.register_buffer("input_std", std, persistent=False)
        self.input_clip = input_clip

        config = SegformerConfig.from_pretrained(base_model)
        config.num_labels = num_classes
        config.num_channels = input_channels
        self.encoder = SegformerModel(config)
        self.decoder = SegformerDecodeHead(config)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in):
        if self.input_mean.numel():
            x_in = (x_in - self.input_mean) / self.input_std
            if self.input_clip is not None:
                x_in = x_in.clamp(-self.input_clip, self.input_clip)

        outputs = self.encoder(x_in,
                               output_attentions = False,
                               output_hidden_states = True,
                               return_dict = True)
        
        logits = self.decoder(outputs.hidden_states)
        x = nn.functional.interpolate(logits, size=(x_in.shape[2], x_in.shape[3]), mode='bilinear', align_corners=False)
        return x, self.softmax(x)
