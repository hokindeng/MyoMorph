#!/usr/bin/env python3
"""
MyoSkeleton MotionGPT Training with Qwen3-0.6B Backbone
Complete retraining pipeline using Qwen3-0.6B instead of T5-Base

Usage:
    # Stage 1: Train VQ-VAE motion tokenizer (same as before)
    python train_myoskeleton_qwen.py --cfg configs/config_myoskeleton_stage1.yaml --stage 1
    
    # Stage 2: Pretrain Qwen3-0.6B with motion tokens
    python train_myoskeleton_qwen.py --cfg configs/config_myoskeleton_qwen_stage2.yaml --stage 2
    
    # Stage 3: Instruction tuning with Qwen3-0.6B
    python train_myoskeleton_qwen.py --cfg configs/config_myoskeleton_qwen_stage3.yaml --stage 3
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mGPT.config import instantiate_from_config, load_config
from mGPT.data.myoskeleton_dataset import MyoSkeletonDataModule
from mGPT.models.mgpt import MotionGPT
from mGPT.utils.myoskeleton_joints import get_simplified_joint_count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_qwen3_requirements():
    """Check if required packages for Qwen3 are installed"""
    try:
        import transformers
        from packaging import version
        
        # Check transformers version
        if version.parse(transformers.__version__) < version.parse("4.51.0"):
            logger.warning(f"⚠️  Transformers version {transformers.__version__} detected.")
            logger.warning("   Qwen3 requires transformers>=4.51.0 for optimal performance.")
            logger.warning("   Consider upgrading: pip install transformers>=4.51.0")
        else:
            logger.info(f"✅ Transformers version {transformers.__version__} is compatible with Qwen3")
            
    except ImportError:
        logger.error("❌ Transformers not found. Install with: pip install transformers>=4.51.0")
        raise


def setup_callbacks(cfg, stage: int):
    """Setup training callbacks for each stage"""
    callbacks = []
    
    # Model checkpointing
    checkpoint_dir = Path(cfg.TEST.CHECKPOINTS)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'myoskeleton_qwen_stage{stage}_' + '{epoch:03d}_{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def train_stage1_vqvae(cfg):
    """Train Stage 1: MyoSkeleton VQ-VAE motion tokenizer (same as original)"""
    logger.info("🔥 Starting MyoSkeleton Stage 1: VQ-VAE Training")
    
    # Data module
    datamodule = instantiate_from_config(cfg.DATASET)
    logger.info(f"Loaded dataset with {datamodule.njoints} joints, {datamodule.nfeats} features")
    
    # Verify correct joint count
    expected_joints = get_simplified_joint_count()
    if datamodule.njoints != expected_joints:
        raise ValueError(f"Expected {expected_joints} joints for MyoSkeleton, got {datamodule.njoints}")
    
    # Model
    model = instantiate_from_config(cfg.model, datamodule=datamodule)
    
    # Logger
    logger_tb = TensorBoardLogger(
        save_dir="./logs",
        name=f"myoskeleton_qwen_stage1",
        version=None
    )
    
    # Callbacks
    callbacks = setup_callbacks(cfg, 1)
    
    # Trainer
    trainer = pl.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        logger=logger_tb,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=cfg.LOGGER.SAVER.LOG_EVERY_STEPS,
        strategy='ddp' if len(cfg.DEVICE) > 1 else None,
        gradient_clip_val=1.0,
        precision='bf16-mixed',  # Use bfloat16 for better efficiency
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    logger.info("✅ Stage 1 VQ-VAE training completed!")
    return trainer.checkpoint_callback.best_model_path


def train_stage2_qwen_pretrain(cfg):
    """Train Stage 2: Qwen3-0.6B pretraining with MyoSkeleton motion tokens"""
    logger.info("🔥 Starting MyoSkeleton Stage 2: Qwen3-0.6B Pretraining")
    
    # Check requirements
    check_qwen3_requirements()
    
    # Data module
    datamodule = instantiate_from_config(cfg.DATASET)
    
    # Model with pretrained VQ-VAE
    model = instantiate_from_config(cfg.model, datamodule=datamodule)
    
    # Load Stage 1 VQ-VAE if specified
    if cfg.TRAIN.PRETRAINED_VAE:
        logger.info(f"Loading pretrained VQ-VAE from: {cfg.TRAIN.PRETRAINED_VAE}")
        vae_checkpoint = torch.load(cfg.TRAIN.PRETRAINED_VAE, map_location='cpu')
        
        # Extract VQ-VAE state dict
        vae_state_dict = {}
        for key, value in vae_checkpoint['state_dict'].items():
            if key.startswith('vae.'):
                vae_state_dict[key[4:]] = value  # Remove 'vae.' prefix
        
        model.vae.load_state_dict(vae_state_dict)
        logger.info("✅ Loaded pretrained VQ-VAE weights")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    qwen_params = sum(p.numel() for p in model.lm.language_model.parameters())
    logger.info(f"📊 Model Statistics:")
    logger.info(f"   - Total parameters: {total_params:,}")
    logger.info(f"   - Qwen3-0.6B backbone: {qwen_params:,}")
    logger.info(f"   - Extended vocabulary: {len(model.lm.tokenizer)} tokens")
    
    # Logger
    logger_tb = TensorBoardLogger(
        save_dir="./logs",
        name=f"myoskeleton_qwen_stage2",
        version=None
    )
    
    # Callbacks
    callbacks = setup_callbacks(cfg, 2)
    
    # Trainer configuration for Qwen3
    trainer = pl.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        logger=logger_tb,
        callbacks=callbacks,
        check_val_every_n_epoch=5,
        log_every_n_steps=cfg.LOGGER.SAVER.LOG_EVERY_STEPS,
        strategy='ddp' if len(cfg.DEVICE) > 1 else None,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        precision='bf16-mixed',  # Use bfloat16 for Qwen3 efficiency
        max_steps=-1,  # Let it run full epochs
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    logger.info("✅ Stage 2 Qwen3-0.6B pretraining completed!")
    return trainer.checkpoint_callback.best_model_path


def train_stage3_qwen_instruct(cfg):
    """Train Stage 3: Qwen3-0.6B instruction tuning for multi-task motion understanding"""
    logger.info("🔥 Starting MyoSkeleton Stage 3: Qwen3-0.6B Instruction Tuning")
    
    # Data module
    datamodule = instantiate_from_config(cfg.DATASET)
    
    # Model
    model = instantiate_from_config(cfg.model, datamodule=datamodule)
    
    # Load Stage 2 pretrained model if specified
    if cfg.TRAIN.PRETRAINED:
        logger.info(f"Loading pretrained model from: {cfg.TRAIN.PRETRAINED}")
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("✅ Loaded pretrained Stage 2 Qwen3 model")
    
    # Logger
    logger_tb = TensorBoardLogger(
        save_dir="./logs",
        name=f"myoskeleton_qwen_stage3",
        version=None
    )
    
    # Callbacks
    callbacks = setup_callbacks(cfg, 3)
    
    # Trainer for instruction tuning
    trainer = pl.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        logger=logger_tb,
        callbacks=callbacks,
        check_val_every_n_epoch=2,
        log_every_n_steps=cfg.LOGGER.SAVER.LOG_EVERY_STEPS,
        strategy='ddp' if len(cfg.DEVICE) > 1 else None,
        gradient_clip_val=0.5,  # Lower gradient clipping for fine-tuning
        accumulate_grad_batches=1,
        precision='bf16-mixed',
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    logger.info("✅ Stage 3 Qwen3-0.6B instruction tuning completed!")
    return trainer.checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser(description="MyoSkeleton MotionGPT Training with Qwen3-0.6B")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3], 
                       help="Training stage (1=VQ-VAE, 2=Qwen3-Pretrain, 3=Qwen3-Instruct)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.cfg)
    
    if args.debug:
        cfg.DEBUG = True
        cfg.TRAIN.NUM_EPOCHS = 5  # Very short debug run
    
    # Set up directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # Print training info
    logger.info("🦴 MyoSkeleton MotionGPT Training with Qwen3-0.6B Backbone")
    logger.info(f"📋 Stage: {args.stage}")
    logger.info(f"📄 Config: {args.cfg}")
    
    # Train based on stage
    if args.stage == 1:
        best_model_path = train_stage1_vqvae(cfg)
    elif args.stage == 2:
        best_model_path = train_stage2_qwen_pretrain(cfg)
    elif args.stage == 3:
        best_model_path = train_stage3_qwen_instruct(cfg)
    
    logger.info(f"🎉 Training completed! Best model saved at: {best_model_path}")
    
    # Print next steps
    if args.stage == 1:
        logger.info("📋 Next step: Update PRETRAINED_VAE in Qwen stage2 config and run:")
        logger.info(f"   python train_myoskeleton_qwen.py --cfg configs/config_myoskeleton_qwen_stage2.yaml --stage 2")
    elif args.stage == 2:
        logger.info("📋 Next step: Update PRETRAINED in Qwen stage3 config and run:")
        logger.info(f"   python train_myoskeleton_qwen.py --cfg configs/config_myoskeleton_qwen_stage3.yaml --stage 3")
    elif args.stage == 3:
        logger.info("🎊 All training stages completed! MyoSkeleton + Qwen3 ready for inference.")
        logger.info("📋 Test the model with:")
        logger.info(f"   python demo_myoskeleton_qwen.py --cfg {args.cfg} --text 'a person walks forward'")
        
    logger.info("💡 Qwen3 Features Available:")
    logger.info("   🧠 Thinking Mode: Enhanced reasoning for complex motion understanding")
    logger.info("   ⚡ Efficiency: 0.6B parameters vs 770M T5-Base (22% smaller)")
    logger.info("   🎯 Advanced: State-of-the-art language capabilities")


if __name__ == "__main__":
    main() 