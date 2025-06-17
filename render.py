#!/usr/bin/env python3
"""
Unified Motion Renderer
Renders motion from various representations like MyoSkeleton or HumanML3D.
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
import mGPT.render.matplot.plot_3d_global as plot_3d
from mGPT.config import parse_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def render_humanml3d(motion_path, output_path):
    """Render HumanML3D motion to a GIF."""
    motion_data = np.load(motion_path)
    if len(motion_data.shape) == 3:
        motion_data = motion_data[None]
    
    plot_3d.draw_to_batch(motion_data, [''], [str(output_path)])
    return output_path


def main():
    # Use the new centralized argument parser
    cfg = parse_args(phase="render")

    motion_path = Path(cfg.RENDER.MOTION_PATH)
    if not motion_path.exists():
        logger.error(f"Motion file not found: {motion_path}")
        return
        
    if cfg.RENDER.OUTPUT:
        output_path = Path(cfg.RENDER.OUTPUT)
    else:
        suffix = ".mp4" if cfg.RENDER.TYPE == "myoskeleton" else ".gif"
        output_path = motion_path.with_suffix(suffix)

    logger.info(f"Rendering motion: {motion_path} ({cfg.RENDER.TYPE})")
    logger.info(f"Output path: {output_path}")

    try:
        if cfg.RENDER.TYPE == "myoskeleton":
            logger.info(f"Render mode: {cfg.RENDER.MYOSKELETON.MODE}")
            result_path = render_myoskeleton_motion(
                motion=str(motion_path),
                output_path=str(output_path),
                render_mode=cfg.RENDER.MYOSKELETON.MODE,
                fps=cfg.RENDER.MYOSKELETON.FPS,
                output_size=tuple(cfg.RENDER.MYOSKELETON.SIZE)
            )
        elif cfg.RENDER.TYPE == "humanml3d":
            result_path = render_humanml3d(motion_path, output_path)
        
        logger.info(f"✅ Successfully rendered: {result_path}")
        
    except Exception as e:
        logger.error(f"❌ Rendering failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 