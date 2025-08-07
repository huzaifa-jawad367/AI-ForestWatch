import torch
import torch.nn as nn
from model.Segforest.Mix_transformer import MixVisionTransformer, mit_b4
from model.Segforest.MFF_blocks import MFFBlocks
from model.Segforest.MultiScale_MultiDecoder import MultiScaleMultiDecoder

class Segforest(nn.Module):
    def __init__(self, 
                 img_size=128, 
                 in_chans=18, 
                 encoder_embed_dims=[64, 128, 320, 512],
                 mff_out_channels=[64, 128, 320],
                 decoder_inner_channels=64,
                 num_classes=3):
        super().__init__()
        # Encoder
        # self.encoder = MixVisionTransformer(
        #     img_size=img_size, 
        #     in_chans=in_chans, 
        #     embed_dims=encoder_embed_dims
        # )
        # Use mit_b4 instead with increased dropout for regularization
        self.encoder = mit_b4(
            img_size=img_size, 
            in_chans=in_chans,
            drop_rate=0.1,  # Add dropout to MLP layers
            drop_path_rate=0.1  # Keep stochastic depth
        )
        # MFF blocks: input channels are the sum of encoder outputs at each scale after concat
        # For MFFBlocks, in_channels_list = [sum of channels after concat for k=1,2,3]
        # Easch concat is 4 encoder outputs resized and concatenated, so sum of encoder_embed_dims
        mff_in_channels = [sum(encoder_embed_dims)] * 3
        self.mff_blocks = MFFBlocks(mff_in_channels, mff_out_channels)
        # Decoder: TB4 channels is encoder_embed_dims[3]
        self.decoder = MultiScaleMultiDecoder(
            mff_channels=mff_out_channels, 
            embed_dims=encoder_embed_dims,
            inner_channels=decoder_inner_channels,
            num_classes=num_classes
        )

    def forward(self, x):
        encoder_outputs = self.encoder(x)  # [TB1, TB2, TB3, TB4]
        mff_outputs = self.mff_blocks(encoder_outputs)  # [MFF_1, MFF_2, MFF_3]
        decoder_outputs = self.decoder(mff_outputs, encoder_outputs)  # [out1, out2, out3]
        
        if self.training:
            # Training mode: return list of 3 upsampled outputs
            upsampled_outputs = [
                torch.nn.functional.interpolate(decoder_outputs[0], scale_factor=4, mode='bilinear', align_corners=False),
                torch.nn.functional.interpolate(decoder_outputs[1], scale_factor=2, mode='bilinear', align_corners=False),
                torch.nn.functional.interpolate(decoder_outputs[2], scale_factor=2, mode='bilinear', align_corners=False)
            ]
            return upsampled_outputs
        else:
            # Evaluation mode: return only the highest resolution output with scale factor 4
            return [torch.nn.functional.interpolate(decoder_outputs[0], scale_factor=4, mode='bilinear', align_corners=False)]