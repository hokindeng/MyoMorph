#!/usr/bin/env python3
"""
Quick MyoSkeleton Motion Renderer
Replaces quick_render.py - renders motion using MyoSkeleton instead of SMPL
"""

import argparse
import numpy as np
from pathlib import Path
import logging
import sys
import os

# Add mGPT to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mGPT.render.myoskeleton_renderer import render_myoskeleton_motion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Quick MyoSkeleton Motion Renderer")
    parser.add_argument("motion_path", type=str, help="Path to motion file (.npy or .h5)")
    parser.add_argument("--output", "-o", type=str, help="Output path for rendered video")
    parser.add_argument("--mode", "-m", type=str, default="skeleton", 
                       choices=["skeleton", "physics", "mesh"],
                       help="Rendering mode")
    parser.add_argument("--fps", type=int, default=20, help="FPS for output video")
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512], 
                       help="Output video size (width height)")
    
    args = parser.parse_args()
    
    # Check if motion file exists
    motion_path = Path(args.motion_path)
    if not motion_path.exists():
        logger.error(f"Motion file not found: {motion_path}")
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = motion_path.parent / f"{motion_path.stem}_myoskeleton.mp4"
    
    logger.info(f"Rendering motion: {motion_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Render mode: {args.mode}")
    
    try:
        # Render motion
        result_path = render_myoskeleton_motion(
            motion=str(motion_path),
            output_path=str(output_path),
            render_mode=args.mode,
            fps=args.fps,
            output_size=tuple(args.size)
        )
        
        logger.info(f"✅ Successfully rendered: {result_path}")
        
    except Exception as e:
        logger.error(f"❌ Rendering failed: {e}")
        raise


if __name__ == "__main__":
    main() 