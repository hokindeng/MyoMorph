import torch
from torch.nn import Module
from omegaconf import OmegaConf
from collections import OrderedDict
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def load_pretrained_model(model: Module, checkpoint_path: str, strict: bool = True) -> Module:
    """
    Load a pre-trained model from a checkpoint file.
    """
    logger.info(f"Loading pre-trained model from: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    
    # Remove the 'model.' prefix if it exists
    clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict, strict=strict)
    
    return model


def load_pretrained_vae(model_with_vae: Module, checkpoint_path: str, strict: bool = True) -> Module:
    """
    Load pre-trained VAE weights into a model that contains a 'vae' attribute.
    """
    logger.info(f"Loading pre-trained VAE from: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")['state_dict']
    
    # Extract only the VAE weights
    vae_weights = {
        k.replace("vae.", ""): v 
        for k, v in state_dict.items() 
        if k.startswith("vae.")
    }

    if not vae_weights:
        raise ValueError(f"No VAE weights (keys starting with 'vae.') found in checkpoint: {checkpoint_path}")
        
    # Load the weights into the model's VAE component
    if hasattr(model_with_vae, 'vae'):
        model_with_vae.vae.load_state_dict(vae_weights, strict=strict)
    else:
        raise AttributeError("The provided model does not have a 'vae' attribute to load weights into.")
    
    return model_with_vae
