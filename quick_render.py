#!/usr/bin/env python3
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to path
sys.path.append(os.getcwd())
import mGPT.render.matplot.plot_3d_global as plot_3d

def quick_render(motion_file, output_name=None):
    """Quickly render a motion file to GIF"""
    
    # Load motion data
    motion_data = np.load(motion_file)
    print(f"📊 Motion shape: {motion_data.shape}")
    print(f"🎬 Motion length: {motion_data.shape[0]} frames")
    
    # Add batch dimension if needed
    if len(motion_data.shape) == 3:
        motion_data = motion_data[None]  # Add batch dimension
    
    # Generate output filename
    if output_name is None:
        motion_path = Path(motion_file)
        output_name = motion_path.stem + "_rendered.gif"
    
    output_path = Path("cache") / output_name
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"🔄 Rendering motion to: {output_path}")
    
    # Render using the same method as the app
    plot_3d.draw_to_batch(motion_data, [''], [str(output_path)])
    
    print(f"✅ Motion rendered successfully!")
    print(f"📁 Output: {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    # Default to the motion file mentioned by user
    motion_file = "cache/motion_20250617_161525.npy"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        motion_file = sys.argv[1]
    
    if not os.path.exists(motion_file):
        print(f"❌ Motion file not found: {motion_file}")
        print("💡 Usage: python quick_render.py [motion_file.npy]")
        sys.exit(1)
    
    quick_render(motion_file) 