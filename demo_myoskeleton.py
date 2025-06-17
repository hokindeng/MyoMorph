#!/usr/bin/env python3
"""
MyoSkeleton MotionGPT Demo Script
Test the retrained model with MyoSkeleton representation
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

from mGPT.config import load_config, instantiate_from_config
from mGPT.data.myoskeleton_dataset import MyoSkeletonDataModule
from mGPT.models.mgpt import MotionGPT
from mGPT.render.myoskeleton_renderer import render_myoskeleton_motion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(cfg, checkpoint_path: str):
    """Load trained MyoSkeleton MotionGPT model"""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    # Data module (needed for model instantiation)
    datamodule = instantiate_from_config(cfg.DATASET)
    
    # Model
    model = instantiate_from_config(cfg.model, datamodule=datamodule)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    logger.info("✅ Model loaded successfully!")
    return model, datamodule


def generate_motion(model, text_prompt: str, max_length: int = 196):
    """Generate motion from text using MyoSkeleton MotionGPT"""
    logger.info(f"Generating motion for: '{text_prompt}'")
    
    # Prepare input
    device = next(model.parameters()).device
    
    # Create input batch
    batch = {
        'text': [text_prompt],
        'length': [max_length]
    }
    
    # Generate motion
    with torch.no_grad():
        # Generate motion tokens
        output = model.forward_test(batch)
        
        # Decode tokens to motion
        if 'motion' in output:
            motion = output['motion'][0]  # First (and only) item in batch
        else:
            # Fallback if different output format
            motion_tokens = output.get('pred_motions', output.get('motion_tokens'))
            motion = model.vae.decode(motion_tokens[0:1])[0]  # Decode first sequence
    
    # Convert to numpy
    if isinstance(motion, torch.Tensor):
        motion = motion.detach().cpu().numpy()
    
    logger.info(f"Generated motion shape: {motion.shape}")
    return motion


def save_motion(motion: np.ndarray, output_path: str):
    """Save generated motion to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, motion)
    logger.info(f"Motion saved to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="MyoSkeleton MotionGPT Demo")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--text", type=str, default="a person walks forward", help="Text prompt for motion generation")
    parser.add_argument("--output", "-o", type=str, help="Output directory for generated motions")
    parser.add_argument("--render", action="store_true", help="Render generated motion as video")
    parser.add_argument("--max_length", type=int, default=196, help="Maximum motion length")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("./results/myoskeleton_demo")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config and model
    cfg = load_config(args.cfg)
    model, datamodule = load_model(cfg, args.checkpoint)
    
    # Generate motion
    try:
        motion = generate_motion(model, args.text, args.max_length)
        
        # Save motion
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        motion_filename = f"motion_{timestamp}.npy"
        motion_path = save_motion(motion, output_dir / motion_filename)
        
        # Create motion info file
        info_path = output_dir / f"motion_{timestamp}.txt"
        with open(info_path, 'w') as f:
            f.write(f"Text Prompt: {args.text}\n")
            f.write(f"Motion Shape: {motion.shape}\n")
            f.write(f"Model Config: {args.cfg}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Generated: {datetime.now()}\n")
        
        logger.info(f"Motion info saved to: {info_path}")
        
        # Render motion if requested
        if args.render:
            logger.info("Rendering motion...")
            video_path = output_dir / f"motion_{timestamp}.mp4"
            
            try:
                rendered_path = render_myoskeleton_motion(
                    motion=motion_path,
                    output_path=str(video_path),
                    render_mode="skeleton"
                )
                logger.info(f"✅ Motion rendered to: {rendered_path}")
            except Exception as e:
                logger.error(f"❌ Rendering failed: {e}")
                logger.info("You can render manually with:")
                logger.info(f"   python quick_render_myoskeleton.py {motion_path}")
        
        # Success summary
        logger.info("🎉 Demo completed successfully!")
        logger.info(f"📁 Results saved in: {output_dir}")
        logger.info(f"🎬 Motion file: {motion_filename}")
        
        if not args.render:
            logger.info("💡 To render the motion, run:")
            logger.info(f"   python quick_render_myoskeleton.py {motion_path}")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 