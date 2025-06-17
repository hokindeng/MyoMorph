#!/usr/bin/env python3
"""
MyoSkeleton MotionGPT Training
Complete retraining pipeline for MotionGPT with MyoSkeleton representation,
supporting multiple language model backbones like T5 and Qwen.

Usage:
    # Stage 1: Train VQ-VAE motion tokenizer (backbone-agnostic)
    python train_myoskeleton.py --cfg configs/config_myoskeleton_stage1.yaml --stage 1
    
    # Stage 2: Pretrain Language Model (e.g., T5 or Qwen)
    # For T5:
    python train_myoskeleton.py --cfg configs/config_myoskeleton_t5_stage2.yaml --stage 2
    # For Qwen:
    python train_myoskeleton.py --cfg configs/config_myoskeleton_qwen_stage2.yaml --stage 2
    
    # Stage 3: Instruction Tuning
    # For T5:
    python train_myoskeleton.py --cfg configs/config_myoskeleton_t5_stage3.yaml --stage 3
    # For Qwen:
    python train_myoskeleton.py --cfg configs/config_myoskeleton_qwen_stage3.yaml --stage 3
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

from mGPT.config import instantiate_from_config, parse_args
from mGPT.data.myoskeleton_dataset import MyoSkeletonDataModule
from mGPT.models.mgpt import MotionGPT
from mGPT.utils.myoskeleton_joints import get_simplified_joint_count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_transformer_requirements(model_name: str):
    """Check if required packages for a given transformer model are installed"""
    if 'qwen' in model_name.lower():
        try:
            import transformers
            from packaging import version
            
            if version.parse(transformers.__version__) < version.parse("4.51.0"):
                logger.warning(f"⚠️  Transformers version {transformers.__version__} detected for Qwen.")
                logger.warning("   Qwen models perform best with transformers>=4.51.0.")
            else:
                logger.info(f"✅ Transformers version {transformers.__version__} is suitable for Qwen.")
                
        except ImportError:
            logger.error("❌ Transformers not found. Install with: pip install transformers>=4.51.0")
            raise


def setup_callbacks(cfg, stage: int, model_name: str):
    """Setup training callbacks for each stage"""
    callbacks = []
    
    # Model checkpointing
    checkpoint_dir = Path(cfg.TEST.CHECKPOINTS)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filename_prefix = f'myoskeleton_{model_name}_stage{stage}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename_prefix + '_{epoch:03d}_{val_loss:.4f}',
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
    """Train Stage 1: MyoSkeleton VQ-VAE motion tokenizer"""
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
        name=f"myoskeleton_stage1_vqvae",
        version=None
    )
    
    # Callbacks
    callbacks = setup_callbacks(cfg, 1, "vqvae")
    
    # Trainer
    trainer = pl.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        logger=logger_tb,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=cfg.LOGGER.SAVER.LOG_EVERY_STEPS,
        strategy='ddp' if len(cfg.DEVICE) > 1 else 'auto',
        gradient_clip_val=1.0,
        precision='bf16-mixed',
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    logger.info("✅ Stage 1 VQ-VAE training completed!")
    return trainer.checkpoint_callback.best_model_path


def train_stage2_lm_pretrain(cfg):
    """Train Stage 2: Language Model pretraining with MyoSkeleton motion tokens"""
    model_name = cfg.model.params.lm.params.model_name
    logger.info(f"🔥 Starting MyoSkeleton Stage 2: {model_name} Pretraining")
    
    # Check requirements if needed
    check_transformer_requirements(model_name)
    
    # Data module
    datamodule = instantiate_from_config(cfg.DATASET)
    
    # Model with pretrained VQ-VAE
    model = instantiate_from_config(cfg.model, datamodule=datamodule)
    
    # Load Stage 1 VQ-VAE if specified
    if cfg.TRAIN.PRETRAINED_VAE:
        logger.info(f"Loading pretrained VQ-VAE from: {cfg.TRAIN.PRETRAINED_VAE}")
        vae_checkpoint = torch.load(cfg.TRAIN.PRETRAINED_VAE, map_location='cpu')
        
        vae_state_dict = {k.replace('vae.', ''): v for k, v in vae_checkpoint['state_dict'].items() if k.startswith('vae.')}
        model.vae.load_state_dict(vae_state_dict)
        logger.info("✅ Loaded pretrained VQ-VAE weights")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    lm_params = sum(p.numel() for p in model.lm.language_model.parameters())
    logger.info(f"📊 Model Statistics:")
    logger.info(f"   - Total parameters: {total_params:,}")
    logger.info(f"   - {model_name} backbone: {lm_params:,}")
    logger.info(f"   - Extended vocabulary: {len(model.lm.tokenizer)} tokens")
    
    # Logger
    logger_tb = TensorBoardLogger(
        save_dir="./logs",
        name=f"myoskeleton_{model_name}_stage2",
        version=None
    )
    
    # Callbacks
    callbacks = setup_callbacks(cfg, 2, model_name)
    
    # Trainer configuration
    trainer = pl.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        logger=logger_tb,
        callbacks=callbacks,
        check_val_every_n_epoch=5,
        log_every_n_steps=cfg.LOGGER.SAVER.LOG_EVERY_STEPS,
        strategy='ddp' if len(cfg.DEVICE) > 1 else 'auto',
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.TRAIN.get("accumulate_grad_batches", 2),
        precision='bf16-mixed',
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    logger.info(f"✅ Stage 2 {model_name} pretraining completed!")
    return trainer.checkpoint_callback.best_model_path


def train_stage3_instruct(cfg):
    """Train Stage 3: Instruction tuning for multi-task motion understanding"""
    model_name = cfg.model.params.lm.params.model_name
    logger.info(f"🔥 Starting MyoSkeleton Stage 3: {model_name} Instruction Tuning")
    
    # Data module
    datamodule = instantiate_from_config(cfg.DATASET)
    
    # Model
    model = instantiate_from_config(cfg.model, datamodule=datamodule)
    
    # Load Stage 2 pretrained model if specified
    if cfg.TRAIN.PRETRAINED:
        logger.info(f"Loading pretrained model from: {cfg.TRAIN.PRETRAINED}")
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(f"✅ Loaded pretrained Stage 2 {model_name} model")
    
    # Logger
    logger_tb = TensorBoardLogger(
        save_dir="./logs",
        name=f"myoskeleton_{model_name}_stage3",
        version=None
    )
    
    # Callbacks
    callbacks = setup_callbacks(cfg, 3, model_name)
    
    # Trainer for instruction tuning
    trainer = pl.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        logger=logger_tb,
        callbacks=callbacks,
        check_val_every_n_epoch=2,
        log_every_n_steps=cfg.LOGGER.SAVER.LOG_EVERY_STEPS,
        strategy='ddp' if len(cfg.DEVICE) > 1 else 'auto',
        gradient_clip_val=cfg.TRAIN.get("gradient_clip_val", 0.5),
        precision='bf16-mixed',
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    logger.info(f"✅ Stage 3 {model_name} instruction tuning completed!")
    return trainer.checkpoint_callback.best_model_path


def main():
    # Use the new centralized argument parser
    cfg = parse_args(phase="train")

    # Set up directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # Determine the model name from the config
    model_name = cfg.model.params.lm.params.get("model_name", "t5")
    
    # Extract stage from config, assuming it's set there
    stage = cfg.TRAIN.get("STAGE", 1)

    # Print training info
    logger.info(f"🦴 MyoSkeleton MotionGPT Training with {model_name.upper()} Backbone")
    logger.info(f"📋 Stage: {stage}")
    logger.info(f"📄 Config: {cfg.get('CONFIG_PATH', 'N/A')}") # Assumes CONFIG_PATH is set
    
    # Train based on stage
    if stage == 1:
        best_model_path = train_stage1_vqvae(cfg)
    elif stage == 2:
        best_model_path = train_stage2_lm_pretrain(cfg)
    elif stage == 3:
        best_model_path = train_stage3_instruct(cfg)
    
    logger.info(f"🎉 Training completed! Best model saved at: {best_model_path}")
    
    # Print next steps
    if stage == 1:
        logger.info("📋 Next step: Update PRETRAINED_VAE in stage 2 config and run pretraining.")
        logger.info(f"   Example: python train_myoskeleton.py --cfg configs/config_myoskeleton_{model_name}_stage2.yaml")
    elif stage == 2:
        logger.info("📋 Next step: Update PRETRAINED in stage 3 config and run instruction tuning.")
        logger.info(f"   Example: python train_myoskeleton.py --cfg configs/config_myoskeleton_{model_name}_stage3.yaml")
    elif stage == 3:
        logger.info(f"🎊 All training stages completed! MyoSkeleton + {model_name.upper()} ready for inference.")
        logger.info("📋 Test the model with:")
        logger.info(f"   python demo.py --cfg {cfg.get('CONFIG_PATH', 'your_config.yaml')} --checkpoint {best_model_path}")


if __name__ == "__main__":
    main() 