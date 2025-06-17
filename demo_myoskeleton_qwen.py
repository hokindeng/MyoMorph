#!/usr/bin/env python3
"""
MyoSkeleton MotionGPT Demo with Qwen3-0.6B Backbone
Test the retrained model with Qwen3-0.6B and MyoSkeleton representation
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_qwen3_availability():
    """Check if Qwen3 model is available"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        logger.info("✅ Qwen3-0.6B is available from Hugging Face")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Qwen3-0.6B not accessible: {e}")
        logger.warning("   Make sure you have internet connection and transformers>=4.51.0")
        return False


def load_model(cfg, checkpoint_path: str):
    """Load trained MyoSkeleton MotionGPT model with Qwen3 backbone"""
    logger.info(f"Loading MyoSkeleton + Qwen3 model from: {checkpoint_path}")
    
    # Check Qwen3 availability
    if not check_qwen3_availability():
        raise RuntimeError("Qwen3-0.6B model not available")
    
    # Data module (needed for model instantiation)
    datamodule = instantiate_from_config(cfg.DATASET)
    
    # Model
    model = instantiate_from_config(cfg.model, datamodule=datamodule)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    qwen_params = sum(p.numel() for p in model.lm.language_model.parameters())
    
    logger.info("🧠 Model Information:")
    logger.info(f"   - Architecture: MyoSkeleton VQ-VAE + Qwen3-0.6B")
    logger.info(f"   - Total parameters: {total_params:,}")
    logger.info(f"   - Qwen3 backbone: {qwen_params:,}")
    logger.info(f"   - Vocabulary size: {len(model.lm.tokenizer)}")
    logger.info(f"   - Motion tokens: {model.lm.m_codebook_size}")
    logger.info(f"   - Thinking mode: {'Enabled' if model.lm.enable_thinking else 'Disabled'}")
    
    logger.info("✅ Model loaded successfully!")
    return model, datamodule


def generate_motion(model, text_prompt: str, max_length: int = 196, enable_thinking: bool = None):
    """Generate motion from text using MyoSkeleton MotionGPT with Qwen3"""
    logger.info(f"🎭 Generating motion for: '{text_prompt}'")
    
    # Set thinking mode if specified
    if enable_thinking is not None:
        original_thinking = model.lm.enable_thinking
        model.lm.enable_thinking = enable_thinking
        logger.info(f"   - Thinking mode: {'Enabled' if enable_thinking else 'Disabled'}")
    
    # Prepare input
    device = next(model.parameters()).device
    
    # Create input batch
    batch = {
        'text': [text_prompt],
        'length': [max_length]
    }
    
    # Generate motion using Qwen3
    try:
        with torch.no_grad():
            # Generate motion tokens using Qwen3's advanced reasoning
            output = model.forward_test(batch)
            
            # Decode tokens to motion
            if 'motion' in output:
                motion = output['motion'][0]  # First (and only) item in batch
            else:
                # Fallback if different output format
                motion_tokens = output.get('pred_motions', output.get('motion_tokens'))
                if motion_tokens is not None:
                    motion = model.vae.decode(motion_tokens[0:1])[0]  # Decode first sequence
                else:
                    raise RuntimeError("No motion output found")
        
        # Convert to numpy
        if isinstance(motion, torch.Tensor):
            motion = motion.detach().cpu().numpy()
        
        logger.info(f"✅ Generated motion shape: {motion.shape}")
        
        # Restore original thinking mode
        if enable_thinking is not None:
            model.lm.enable_thinking = original_thinking
            
        return motion
        
    except Exception as e:
        logger.error(f"❌ Motion generation failed: {e}")
        # Restore original thinking mode
        if enable_thinking is not None:
            model.lm.enable_thinking = original_thinking
        raise


def test_thinking_modes(model, text_prompt: str):
    """Test both thinking and non-thinking modes of Qwen3"""
    logger.info("🧠 Testing Qwen3 Thinking Modes:")
    
    results = {}
    
    # Test with thinking enabled
    logger.info("   Testing with THINKING mode...")
    try:
        motion_thinking = generate_motion(model, text_prompt, enable_thinking=True)
        results['thinking'] = {
            'motion': motion_thinking,
            'shape': motion_thinking.shape,
            'success': True
        }
        logger.info(f"   ✅ Thinking mode: {motion_thinking.shape}")
    except Exception as e:
        results['thinking'] = {'success': False, 'error': str(e)}
        logger.error(f"   ❌ Thinking mode failed: {e}")
    
    # Test with thinking disabled
    logger.info("   Testing with NON-THINKING mode...")
    try:
        motion_direct = generate_motion(model, text_prompt, enable_thinking=False)
        results['non_thinking'] = {
            'motion': motion_direct,
            'shape': motion_direct.shape,
            'success': True
        }
        logger.info(f"   ✅ Non-thinking mode: {motion_direct.shape}")
    except Exception as e:
        results['non_thinking'] = {'success': False, 'error': str(e)}
        logger.error(f"   ❌ Non-thinking mode failed: {e}")
    
    return results


def save_motion(motion: np.ndarray, output_path: str, metadata: dict = None):
    """Save generated motion to file with metadata"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save motion
    np.save(output_path, motion)
    
    # Save metadata if provided
    if metadata:
        metadata_path = output_path.with_suffix('.txt')
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
    
    logger.info(f"💾 Motion saved to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="MyoSkeleton MotionGPT Demo with Qwen3-0.6B")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--text", type=str, default="a person walks forward", help="Text prompt for motion generation")
    parser.add_argument("--output", "-o", type=str, help="Output directory for generated motions")
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3 thinking mode")
    parser.add_argument("--no-thinking", action="store_true", help="Disable Qwen3 thinking mode")
    parser.add_argument("--test-modes", action="store_true", help="Test both thinking and non-thinking modes")
    parser.add_argument("--max_length", type=int, default=196, help="Maximum motion length")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("./results/myoskeleton_qwen_demo")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine thinking mode
    enable_thinking = None
    if args.thinking:
        enable_thinking = True
    elif args.no_thinking:
        enable_thinking = False
    
    # Load config and model
    cfg = load_config(args.cfg)
    model, datamodule = load_model(cfg, args.checkpoint)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if args.test_modes:
            # Test both thinking modes
            logger.info("🔬 Testing both thinking and non-thinking modes...")
            results = test_thinking_modes(model, args.text)
            
            # Save results for both modes
            for mode, result in results.items():
                if result['success']:
                    motion_filename = f"motion_{mode}_{timestamp}.npy"
                    metadata = {
                        "Text Prompt": args.text,
                        "Mode": mode,
                        "Motion Shape": str(result['shape']),
                        "Model": "MyoSkeleton + Qwen3-0.6B",
                        "Config": args.cfg,
                        "Checkpoint": args.checkpoint,
                        "Generated": datetime.now().isoformat()
                    }
                    save_motion(result['motion'], output_dir / motion_filename, metadata)
            
        else:
            # Single generation
            motion = generate_motion(model, args.text, args.max_length, enable_thinking)
            
            # Determine mode name for filename
            if enable_thinking is True:
                mode_name = "thinking"
            elif enable_thinking is False:
                mode_name = "direct"
            else:
                mode_name = "default"
            
            # Save motion
            motion_filename = f"motion_{mode_name}_{timestamp}.npy"
            metadata = {
                "Text Prompt": args.text,
                "Thinking Mode": str(enable_thinking),
                "Motion Shape": str(motion.shape),
                "Model": "MyoSkeleton + Qwen3-0.6B",
                "Config": args.cfg,
                "Checkpoint": args.checkpoint,
                "Generated": datetime.now().isoformat()
            }
            
            motion_path = save_motion(motion, output_dir / motion_filename, metadata)
            
            logger.info("🎉 Demo completed successfully!")
            logger.info(f"📁 Results saved in: {output_dir}")
            logger.info(f"🎬 Motion file: {motion_filename}")
            
            logger.info("💡 To render the motion:")
            logger.info(f"   python quick_render_myoskeleton.py {motion_path}")
        
        # Print Qwen3 advantages
        logger.info("🌟 Qwen3-0.6B Advantages:")
        logger.info("   🧠 Advanced reasoning with thinking mode")
        logger.info("   ⚡ 22% fewer parameters than T5-Base (0.6B vs 0.77B)")
        logger.info("   🎯 State-of-the-art language understanding")
        logger.info("   🔄 Seamless thinking/non-thinking mode switching")
        logger.info("   🚀 Optimized for motion-language tasks")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 