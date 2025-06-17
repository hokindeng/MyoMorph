"""
MyoSkeleton VQ-VAE Architecture
Updated to handle MyoSkeleton joint representation (28 simplified joints = 84 features)
Replaces the original VQ-VAE that was designed for 22-joint HumanML3D format
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union

from mGPT.archs.tools.quantize_cnn import QuantizeEMAReset, QuantizeReset
from mGPT.utils.myoskeleton_joints import (
    myoskeleton_simplified_joints, 
    get_simplified_joint_count,
    myoskeleton_joints_info
)


class MyoSkeletonVQVAE(nn.Module):
    """
    VQ-VAE for MyoSkeleton motion representation
    Handles 28 simplified joints (84 features) instead of 22 joints (66 features)
    """
    
    def __init__(
        self,
        nfeats: int = 84,  # 28 joints * 3 = 84 features for MyoSkeleton
        quantizer: str = 'ema_reset',
        code_dim: int = 512,
        nb_code: int = 512,
        mu: float = 0.99,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = 'relu',
        norm: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        self.nfeats = nfeats
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.down_t = down_t
        self.stride_t = stride_t
        
        # Verify we're using correct number of features for MyoSkeleton
        expected_nfeats = get_simplified_joint_count() * 3
        if nfeats != expected_nfeats:
            raise ValueError(f"Expected {expected_nfeats} features for MyoSkeleton, got {nfeats}")
        
        # Activation function
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'gelu':
            activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Normalization
        if norm == 'batch':
            norm_fn = nn.BatchNorm1d
        elif norm == 'layer':
            norm_fn = nn.LayerNorm
        else:
            norm_fn = None
        
        # Encoder - downsample temporal dimension and learn features
        self.encoder = MyoSkeletonEncoder(
            nfeats=nfeats,
            width=width,
            depth=depth,
            down_t=down_t,
            stride_t=stride_t,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation_fn,
            norm=norm_fn
        )
        
        # Quantizer
        if quantizer == 'ema_reset':
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
        elif quantizer == 'reset':
            self.quantizer = QuantizeReset(nb_code, code_dim)
        else:
            raise ValueError(f"Unsupported quantizer: {quantizer}")
        
        # Decoder - upsample and reconstruct motion
        self.decoder = MyoSkeletonDecoder(
            nfeats=nfeats,
            width=width,
            depth=depth,
            down_t=down_t,
            stride_t=stride_t,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation_fn,
            norm=norm_fn
        )
    
    def encode(self, motion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode motion to discrete tokens
        
        Args:
            motion: (batch_size, seq_len, nfeats) MyoSkeleton motion
            
        Returns:
            codes: (batch_size, seq_len//down_t) discrete motion tokens
            quantized: (batch_size, seq_len//down_t, code_dim) quantized features
        """
        # Encode motion to continuous features
        encoded = self.encoder(motion)  # (B, T//down_t, width)
        
        # Quantize to discrete codes
        quantized, codes, commit_loss = self.quantizer(encoded)
        
        return codes, quantized
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete tokens back to motion
        
        Args:
            codes: (batch_size, seq_len//down_t) discrete motion tokens
            
        Returns:
            motion: (batch_size, seq_len, nfeats) reconstructed MyoSkeleton motion
        """
        # Dequantize codes to continuous features
        quantized = self.quantizer.dequantize(codes)
        
        # Decode to motion
        motion = self.decoder(quantized)
        
        return motion
    
    def forward(self, motion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VQ-VAE forward pass
        
        Args:
            motion: (batch_size, seq_len, nfeats) MyoSkeleton motion
            
        Returns:
            reconstructed_motion: (batch_size, seq_len, nfeats)
            commit_loss: VQ commitment loss
            perplexity: Codebook usage measure
        """
        # Encode
        encoded = self.encoder(motion)
        
        # Quantize
        quantized, commit_loss, perplexity = self.quantizer(encoded)
        
        # Decode
        reconstructed_motion = self.decoder(quantized)
        
        return reconstructed_motion, commit_loss, perplexity


class MyoSkeletonEncoder(nn.Module):
    """
    Encoder for MyoSkeleton motion - converts motion sequences to latent features
    """
    
    def __init__(
        self,
        nfeats: int,
        width: int,
        depth: int,
        down_t: int,
        stride_t: int,
        dilation_growth_rate: int,
        activation: nn.Module,
        norm: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.nfeats = nfeats
        self.width = width
        self.down_t = down_t
        
        # Input projection
        self.input_proj = nn.Linear(nfeats, width)
        
        # Temporal downsampling layers
        layers = []
        current_width = width
        
        for i in range(depth):
            # Temporal convolution with dilation
            dilation = dilation_growth_rate ** i
            
            conv_layer = nn.Conv1d(
                current_width, current_width,
                kernel_size=3,
                stride=stride_t if i == 0 else 1,  # Only downsample on first layer
                dilation=dilation,
                padding=dilation,
                groups=1
            )
            
            layers.append(conv_layer)
            
            if norm:
                layers.append(norm(current_width))
            
            layers.append(activation)
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Linear(current_width, width)
    
    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion: (batch_size, seq_len, nfeats)
        Returns:
            encoded: (batch_size, seq_len//down_t, width)
        """
        batch_size, seq_len, _ = motion.shape
        
        # Project input features
        x = self.input_proj(motion)  # (B, T, width)
        
        # Transpose for conv1d: (B, width, T)
        x = x.transpose(1, 2)
        
        # Apply temporal convolutions
        x = self.conv_layers(x)  # (B, width, T//down_t)
        
        # Transpose back: (B, T//down_t, width)
        x = x.transpose(1, 2)
        
        # Output projection
        encoded = self.output_proj(x)
        
        return encoded


class MyoSkeletonDecoder(nn.Module):
    """
    Decoder for MyoSkeleton motion - converts latent features back to motion sequences
    """
    
    def __init__(
        self,
        nfeats: int,
        width: int,
        depth: int,
        down_t: int,
        stride_t: int,
        dilation_growth_rate: int,
        activation: nn.Module,
        norm: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.nfeats = nfeats
        self.width = width
        self.down_t = down_t
        self.stride_t = stride_t
        
        # Input projection
        self.input_proj = nn.Linear(width, width)
        
        # Temporal upsampling layers (reverse of encoder)
        layers = []
        current_width = width
        
        for i in range(depth):
            # Temporal transpose convolution for upsampling
            if i == depth - 1:  # Last layer upsamples
                conv_layer = nn.ConvTranspose1d(
                    current_width, current_width,
                    kernel_size=stride_t * 2,
                    stride=stride_t,
                    padding=stride_t // 2
                )
            else:
                dilation = dilation_growth_rate ** (depth - 1 - i)
                conv_layer = nn.Conv1d(
                    current_width, current_width,
                    kernel_size=3,
                    dilation=dilation,
                    padding=dilation
                )
            
            layers.append(conv_layer)
            
            if norm:
                layers.append(norm(current_width))
            
            if i < depth - 1:  # No activation on last layer
                layers.append(activation)
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Output projection to motion features
        self.output_proj = nn.Linear(current_width, nfeats)
    
    def forward(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Args:
            quantized: (batch_size, seq_len//down_t, width)
        Returns:
            motion: (batch_size, seq_len, nfeats)
        """
        # Project input
        x = self.input_proj(quantized)  # (B, T//down_t, width)
        
        # Transpose for conv1d: (B, width, T//down_t)
        x = x.transpose(1, 2)
        
        # Apply temporal convolutions
        x = self.conv_layers(x)  # (B, width, T)
        
        # Transpose back: (B, T, width)
        x = x.transpose(1, 2)
        
        # Project to motion features
        motion = self.output_proj(x)  # (B, T, nfeats)
        
        return motion


# Factory function for easy instantiation
def create_myoskeleton_vqvae(**kwargs) -> MyoSkeletonVQVAE:
    """Create MyoSkeleton VQ-VAE with default parameters"""
    default_params = {
        'nfeats': get_simplified_joint_count() * 3,  # 28 joints * 3 = 84
        'quantizer': 'ema_reset',
        'code_dim': 512,
        'nb_code': 512,
        'mu': 0.99,
        'down_t': 2,
        'stride_t': 2,
        'width': 512,
        'depth': 3,
        'dilation_growth_rate': 3,
        'activation': 'relu',
        'norm': None
    }
    
    # Update with provided kwargs
    default_params.update(kwargs)
    
    return MyoSkeletonVQVAE(**default_params) 