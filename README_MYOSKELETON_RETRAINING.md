# 🦴 MyoSkeleton MotionGPT Retraining Guide

Complete retraining of MotionGPT with **MyoSkeleton representation** instead of SMPL/HumanML3D format.

## 🎯 Overview

This guide replaces the original 22-joint HumanML3D representation with your **28-joint MyoSkeleton model**, requiring complete retraining of:

1. **VQ-VAE Motion Tokenizer** (Stage 1)
2. **T5 Language Model Pretraining** (Stage 2) 
3. **Instruction Tuning** (Stage 3)

## 📊 Key Changes

| Component | Original | MyoSkeleton |
|-----------|----------|-------------|
| **Joint Count** | 22 joints | 28 simplified joints |
| **Features** | 66 features (22×3) | 84 features (28×3) |
| **Motion Format** | HumanML3D | MyoSkeleton H5 |
| **Visualization** | SMPL fitting | Native MyoSkeleton |
| **Motion Tokens** | 512 codes | 512 codes (retrained) |

## 🚀 Quick Start

### 1. Prepare Your MyoSkeleton Dataset

```bash
# Create dataset directory
mkdir -p datasets/myoskeleton_h5/

# Place your H5 files with MyoSkeleton motion data:
# datasets/myoskeleton_h5/
# ├── train_motions.h5      # Training data
# ├── val_motions.h5        # Validation data  
# └── test_motions.h5       # Test data
```

**H5 File Format Expected:**
```python
# Each H5 file should contain:
{
    'motion': np.array,     # Shape: (n_sequences, max_length, 84)  # 28 joints × 3
    'text': list,           # Text descriptions for each motion
    'lengths': np.array,    # Actual length of each motion sequence
}
```

### 2. Three-Stage Training Process

#### **Stage 1: Train MyoSkeleton VQ-VAE**
```bash
# Train motion tokenizer from scratch
python train_myoskeleton.py --cfg configs/config_myoskeleton_stage1.yaml --stage 1

# Expected training time: ~2-3 days on single GPU
# Output: ./checkpoints/myoskeleton_stage1/latest.ckpt
```

#### **Stage 2: Pretrain T5 with MyoSkeleton Tokens**
```bash
# Update config with Stage 1 checkpoint path
# Edit configs/config_myoskeleton_stage2.yaml:
# TRAIN.PRETRAINED_VAE: ./checkpoints/myoskeleton_stage1/latest.ckpt

python train_myoskeleton.py --cfg configs/config_myoskeleton_stage2.yaml --stage 2

# Expected training time: ~3-4 days on single GPU  
# Output: ./checkpoints/myoskeleton_stage2/latest.ckpt
```

#### **Stage 3: Instruction Tuning**
```bash
# Update config with Stage 2 checkpoint path
# Edit configs/config_myoskeleton_stage3.yaml:
# TRAIN.PRETRAINED: ./checkpoints/myoskeleton_stage2/latest.ckpt

python train_myoskeleton.py --cfg configs/config_myoskeleton_stage3.yaml --stage 3

# Expected training time: ~2-3 days on single GPU
# Output: ./checkpoints/myoskeleton_stage3/latest.ckpt
```

### 3. Test Your Retrained Model

```bash
# Generate motion from text
python demo_myoskeleton.py \
    --cfg configs/config_myoskeleton_stage3.yaml \
    --checkpoint ./checkpoints/myoskeleton_stage3/latest.ckpt \
    --text "a person walks forward" \
    --render

# Output: ./results/myoskeleton_demo/motion_TIMESTAMP.npy
#         ./results/myoskeleton_demo/motion_TIMESTAMP.mp4
```

## 📁 File Structure

```
MyoMorph/
├── configs/
│   ├── config_myoskeleton_stage1.yaml    # VQ-VAE training config
│   ├── config_myoskeleton_stage2.yaml    # T5 pretraining config  
│   └── config_myoskeleton_stage3.yaml    # Instruction tuning config
├── mGPT/
│   ├── archs/
│   │   └── mgpt_vae.py                   # Updated VQ-VAE for MyoSkeleton
│   ├── data/
│   │   └── myoskeleton_dataset.py        # MyoSkeleton data loader
│   ├── utils/
│   │   └── myoskeleton_joints.py         # Joint definitions & mappings
│   └── render/
│       └── myoskeleton_renderer.py       # MyoSkeleton visualization
├── datasets/
│   └── myoskeleton_h5/                   # Your H5 motion datasets
├── train_myoskeleton.py                  # Main training script
├── demo_myoskeleton.py                   # Demo/inference script
└── quick_render_myoskeleton.py           # Quick motion rendering
```

## 🔧 Configuration Details

### Stage 1: VQ-VAE Configuration
- **Input Features**: 84 (28 joints × 3 coordinates)
- **Codebook Size**: 512 motion tokens
- **Downsampling**: 4x temporal compression (20fps → 5fps)
- **Architecture**: Encoder-Quantizer-Decoder with dilated convolutions

### Stage 2: T5 Pretraining
- **Model**: T5-Base (770M parameters)
- **Vocabulary**: 32,100 (T5) + 512 (motion tokens) + 3 (special) = 32,615
- **Training Tasks**: Motion-text pairs with template instructions
- **Frozen VQ-VAE**: Uses pretrained Stage 1 tokenizer

### Stage 3: Instruction Tuning
- **Multi-task Learning**: Text-to-motion, motion-to-text, prediction, in-between
- **Template Instructions**: Guided task-specific fine-tuning
- **Lower Learning Rate**: 5e-5 for stable fine-tuning

## 📈 Training Progress Monitoring

```bash
# Monitor training with TensorBoard
tensorboard --logdir ./logs

# Key metrics to watch:
# Stage 1: reconstruction_loss, commit_loss, perplexity
# Stage 2: language_model_loss, motion_reconstruction  
# Stage 3: task_specific_losses, validation_metrics
```

## 🎛️ Advanced Options

### Debug Mode (Quick Testing)
```bash
# Short training runs for testing
python train_myoskeleton.py --cfg CONFIG_FILE --stage STAGE --debug
```

### Multi-GPU Training
```bash
# Update config DEVICE: [0, 1, 2, 3] for 4-GPU training
# Automatically uses DDP strategy
```

### Custom Joint Configuration
Edit `mGPT/utils/myoskeleton_joints.py` to modify:
- Joint names and hierarchy
- Simplified joint set for motion generation
- Motion constraints and limits

## 🔍 Troubleshooting

### Common Issues

1. **Dataset Loading Errors**
   ```bash
   # Check H5 file structure
   python -c "import h5py; f=h5py.File('your_file.h5', 'r'); print(list(f.keys()))"
   ```

2. **Memory Issues** 
   - Reduce batch size in configs
   - Enable gradient accumulation
   - Use mixed precision training

3. **Joint Count Mismatch**
   - Verify your H5 data has 84 features (28 joints × 3)
   - Check `myoskeleton_simplified_joints` list

### Performance Optimization

```bash
# Stage 1: Focus on reconstruction quality
# Monitor: MPJPE, reconstruction loss, codebook usage

# Stage 2: Language-motion alignment  
# Monitor: Cross-modal reconstruction, text-motion consistency

# Stage 3: Task performance
# Monitor: FID, R-precision, multi-task metrics
```

## 🎊 Expected Results

After complete retraining, your MyoSkeleton MotionGPT should:

✅ **Generate anatomically realistic motions** using MyoSkeleton representation  
✅ **Understand natural language** motion descriptions  
✅ **Support multiple tasks**: text-to-motion, motion-to-text, prediction, in-between  
✅ **Visualize directly** without SMPL dependency  
✅ **Leverage MyoSkeleton physics** for realistic motion synthesis  

## 📚 Next Steps

1. **Collect More Data**: Expand your MyoSkeleton motion dataset
2. **Fine-tune Tasks**: Specialize for specific motion domains  
3. **Integration**: Connect with your `myo_api` for real-time control
4. **Evaluation**: Compare against original MotionGPT on your metrics

---

## 🆘 Support

If you encounter issues during retraining:

1. Check logs in `./logs/` directory
2. Verify dataset format and paths
3. Ensure sufficient GPU memory (16GB+ recommended)
4. Monitor training curves for convergence

**Happy Training!** 🚀🦴 