#!/usr/bin/env python3
"""
MotionGPT Unified Demo Script
Test a trained MotionGPT model with any supported backbone and representation.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mGPT.config import parse_args, instantiate_from_config
from mGPT.render.myoskeleton_renderer import render_myoskeleton_motion
from mGPT.render.matplot.plot_3d_global import draw_to_batch as render_humanml3d_motion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(cfg, checkpoint_path: str):
    """Load a trained MotionGPT model from a checkpoint."""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    # Instantiate the datamodule from config to get dataset info
    datamodule = instantiate_from_config(cfg.DATASET)
    
    # Instantiate the model
    model = instantiate_from_config(cfg.model, datamodule=datamodule)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    logger.info("✅ Model loaded successfully!")
    return model, datamodule


def generate_motion(model, text_prompt: str, max_length: int = 196):
    """Generate motion from a text prompt using the loaded model."""
    logger.info(f"Generating motion for: '{text_prompt}'")
    
    device = next(model.parameters()).device
    
    # Prepare a batch for inference
    batch = {'text': [text_prompt], 'length': [max_length]}
    
    with torch.no_grad():
        # The model's forward_test method handles the generation logic
        output = model.forward_test(batch)
        
        # The output format can vary, so we handle different keys
        if 'motion' in output:
            motion = output['motion'][0]
        else:
            motion_tokens = output.get('pred_motions', output.get('motion_tokens'))
            motion = model.vae.decode(motion_tokens[0:1])[0]
            
    motion_np = motion.detach().cpu().numpy()
    
    logger.info(f"Generated motion of shape: {motion_np.shape}")
    return motion_np


def save_motion(motion: np.ndarray, text_prompt: str, cfg, checkpoint_path: str, output_dir: Path):
    """Save the generated motion and accompanying metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = cfg.model.params.lm.params.get("model_name", "model")
    
    # Create a unique filename
    motion_filename = f"{model_name}_{timestamp}.npy"
    motion_path = output_dir / motion_filename
    
    # Save the motion array
    np.save(motion_path, motion)
    logger.info(f"Motion saved to: {motion_path}")
    
    # Save metadata to a text file
    info_path = output_dir / f"{model_name}_{timestamp}.txt"
    with open(info_path, 'w') as f:
        f.write(f"Text Prompt: {text_prompt}\n")
        f.write(f"Motion Shape: {motion.shape}\n")
        f.write(f"Model Config: {cfg.CONFIG_PATH}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Generated On: {datetime.now()}\n")
        
    logger.info(f"Motion metadata saved to: {info_path}")
    return motion_path


def render_motion(motion_path: str, datamodule, output_path: str):
    """Render the motion to a video based on the dataset type."""
    dataset_name = datamodule.__class__.__name__.lower()
    logger.info(f"Rendering motion for '{dataset_name}' dataset type...")
    
    try:
        if "myoskeleton" in dataset_name:
            # Use the MyoSkeleton renderer
            rendered_path = render_myoskeleton_motion(
                motion=motion_path,
                output_path=output_path,
                render_mode="skeleton"
            )
        elif "humanml3d" in dataset_name:
            # Use the HumanML3D renderer
            motion_data = np.load(motion_path)
            render_humanml3d_motion(motion_data, [''], [output_path])
            rendered_path = output_path
        else:
            logger.warning(f"No specific renderer for '{dataset_name}'. Skipping video generation.")
            return None
            
        logger.info(f"✅ Motion rendered successfully to: {rendered_path}")
        return rendered_path
    except Exception as e:
        logger.error(f"❌ Rendering failed: {e}")
        logger.info("You may be able to render it manually using one of the quick_render scripts.")
        return None


def main():
    # Use the new centralized argument parser
    cfg = parse_args(phase="demo")
    
    output_dir = Path(cfg.DEMO.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store the original config path for metadata
    cfg.CONFIG_PATH = cfg.DEMO.CFG
    
    # Load the configuration and model
    model, datamodule = load_model(cfg, cfg.DEMO.CHECKPOINT)
    
    try:
        # Generate and save the motion
        motion_np = generate_motion(model, cfg.DEMO.TEXT, cfg.DEMO.MAX_LENGTH)
        motion_path = save_motion(motion_np, cfg.DEMO.TEXT, cfg, cfg.DEMO.CHECKPOINT, output_dir)
        
        # Render if requested
        if cfg.DEMO.RENDER:
            video_path = motion_path.with_suffix(".mp4")
            render_motion(str(motion_path), datamodule, str(video_path))
        
        logger.info("🎉 Demo finished successfully!")
        logger.info(f"📁 Outputs are saved in: {output_dir.resolve()}")
        if not cfg.DEMO.RENDER:
            logger.info("💡 To render the output, add the --render flag to your config or command line.")
            
    except Exception as e:
        logger.error(f"❌ An error occurred during the demo: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 