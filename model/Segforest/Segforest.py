import torch
import torch.nn as nn
from model.Segforest.Mix_transformer import MixVisionTransformer
from model.Segforest.MFF_blocks import MFFBlocks
from model.Segforest.MultiScale_MultiDecoder import MultiScaleMultiDecoder

class Segforest(nn.Module):
    def __init__(self, 
                 img_size=128, 
                 in_chans=18, 
                 encoder_embed_dims=[64, 128, 320, 512],
                 mff_out_channels=128,
                 decoder_inner_channels=64,
                 num_classes=3):
        super().__init__()
        # Encoder
        self.encoder = MixVisionTransformer(
            img_size=img_size, 
            in_chans=in_chans, 
            embed_dims=encoder_embed_dims
        )
        # MFF blocks: input channels are the sum of encoder outputs at each scale after concat
        # For MFFBlocks, in_channels_list = [sum of channels after concat for k=1,2,3]
        # Each concat is 4 encoder outputs resized and concatenated, so sum of encoder_embed_dims
        mff_in_channels = [sum(encoder_embed_dims)] * 3
        self.mff_blocks = MFFBlocks(mff_in_channels, mff_out_channels)
        # Decoder: TB4 channels is encoder_embed_dims[3]
        self.decoder = MultiScaleMultiDecoder(
            mff_channels=[mff_out_channels]*3, 
            tb4_channels=encoder_embed_dims[3],
            inner_channels=decoder_inner_channels,
            num_classes=num_classes
        )

    def forward(self, x):
        encoder_outputs = self.encoder(x)  # [TB1, TB2, TB3, TB4]
        mff_outputs = self.mff_blocks(encoder_outputs)  # [MFF_1, MFF_2, MFF_3]
        decoder_outputs = self.decoder(mff_outputs, encoder_outputs)  # [out1, out2, out3]
        return decoder_outputs
