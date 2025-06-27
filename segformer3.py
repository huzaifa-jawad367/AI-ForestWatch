import torch.nn as nn

from transformers import SegformerForSemanticSegmentation

class CustomSegformer(nn.Module):
    def __init__(self, config, size = 128):
        super().__init__()

        self.model = SegformerForSemanticSegmentation(config)
        self.softmax = nn.Softmax(dim=1)
        self.output_size = size

    def forward(self, x_in):
        out = self.model(x_in)
        logits = out.logits
        x = nn.functional.interpolate(logits, size=(self.output_size, self.output_size), mode='bilinear')
        return x, self.softmax(x)