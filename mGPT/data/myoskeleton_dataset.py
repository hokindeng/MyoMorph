"""
MyoSkeleton Dataset Loader for H5 files
Replaces SMPL-based data loading with native MyoSkeleton motion data
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import random
import logging

from mGPT.utils.myoskeleton_joints import (
    myoskeleton_joints, myoskeleton_simplified_joints, 
    myoskeleton_joints_info, get_simplified_joint_count
)

logger = logging.getLogger(__name__)


class MyoSkeletonDataset(Dataset):
    """
    Dataset for MyoSkeleton motion data stored in H5 format
    """
    
    def __init__(
        self, 
        data_root: str,
        split: str = "train",
        use_simplified_joints: bool = True,
        min_motion_len: int = 40,
        max_motion_len: int = 196,
        unit_length: int = 4,
        fps: int = 20,
        **kwargs
    ):
        """
        Args:
            data_root: Path to directory containing H5 files
            split: Dataset split (train/val/test) 
            use_simplified_joints: Use simplified joint set for motion generation
            min_motion_len: Minimum motion sequence length
            max_motion_len: Maximum motion sequence length
            unit_length: Unit length for motion tokenization (downsampling factor)
            fps: Original FPS of motion data
        """
        self.data_root = Path(data_root)
        self.split = split
        self.use_simplified_joints = use_simplified_joints
        self.min_motion_len = min_motion_len
        self.max_motion_len = max_motion_len
        self.unit_length = unit_length
        self.fps = fps
        
        # Joint configuration
        if use_simplified_joints:
            self.joints = myoskeleton_simplified_joints
            self.njoints = get_simplified_joint_count()
        else:
            self.joints = myoskeleton_joints
            self.njoints = len(myoskeleton_joints)
            
        self.nfeats = self.njoints * 3  # x, y, z coordinates
        
        # Load data
        self.motion_data = []
        self.text_data = []
        self.length_data = []
        self._load_h5_data()
        
        logger.info(f"Loaded {len(self.motion_data)} MyoSkeleton motion sequences for {split}")
        
    def _load_h5_data(self):
        """Load MyoSkeleton motion data from H5 files"""
        h5_files = list(self.data_root.glob(f"{self.split}*.h5"))
        
        if not h5_files:
            h5_files = list(self.data_root.glob("*.h5"))
            logger.warning(f"No split-specific H5 files found, using all H5 files")
        
        for h5_file in h5_files:
            logger.info(f"Loading {h5_file}")
            
            with h5py.File(h5_file, 'r') as f:
                # Extract motion data 
                if 'motion' in f:
                    motions = f['motion'][:]
                elif 'poses' in f:
                    motions = f['poses'][:]
                else:
                    logger.error(f"No motion data found in {h5_file}")
                    continue
                    
                # Extract text descriptions
                if 'text' in f:
                    texts = [text.decode() if isinstance(text, bytes) else text 
                            for text in f['text'][:]]
                elif 'captions' in f:
                    texts = [text.decode() if isinstance(text, bytes) else text 
                            for text in f['captions'][:]]
                else:
                    # Generate placeholder texts if none available
                    texts = [f"motion_{i}" for i in range(len(motions))]
                    logger.warning(f"No text data found in {h5_file}, using placeholders")
                
                # Extract lengths if available
                if 'lengths' in f:
                    lengths = f['lengths'][:]
                else:
                    lengths = [len(motion) for motion in motions]
                
                # Process and filter motions
                for i, (motion, text, length) in enumerate(zip(motions, texts, lengths)):
                    # Skip motions that are too short or too long
                    if length < self.min_motion_len or length > self.max_motion_len:
                        continue
                        
                    # Process motion data
                    processed_motion = self._process_motion(motion)
                    if processed_motion is not None:
                        self.motion_data.append(processed_motion)
                        self.text_data.append(text)
                        self.length_data.append(length)
    
    def _process_motion(self, motion: np.ndarray) -> Optional[np.ndarray]:
        """Process raw motion data for MyoSkeleton"""
        try:
            # Handle different motion data formats
            if motion.ndim == 3:  # (frames, joints, 3)
                if motion.shape[1] != self.njoints:
                    logger.warning(f"Motion has {motion.shape[1]} joints, expected {self.njoints}")
                    return None
                # Reshape to (frames, features)
                motion = motion.reshape(motion.shape[0], -1)
            elif motion.ndim == 2:  # (frames, features)
                if motion.shape[1] != self.nfeats:
                    logger.warning(f"Motion has {motion.shape[1]} features, expected {self.nfeats}")
                    return None
            else:
                logger.error(f"Unexpected motion shape: {motion.shape}")
                return None
            
            # Normalize motion (optional - can be dataset specific)
            motion = self._normalize_motion(motion)
            
            return motion.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error processing motion: {e}")
            return None
    
    def _normalize_motion(self, motion: np.ndarray) -> np.ndarray:
        """Normalize motion data (implement your specific normalization)"""
        # Example normalization - adjust based on your data
        # Center around pelvis
        pelvis_idx = myoskeleton_joints_info.get("pelvis", 0) * 3
        if pelvis_idx < motion.shape[1]:
            pelvis_pos = motion[:, pelvis_idx:pelvis_idx+3]
            # Center motion around pelvis
            motion[:, ::3] -= pelvis_pos[:, 0:1]  # x coordinates
            motion[:, 1::3] -= pelvis_pos[:, 1:2]  # y coordinates
            motion[:, 2::3] -= pelvis_pos[:, 2:3]  # z coordinates
        
        return motion
    
    def __len__(self) -> int:
        return len(self.motion_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        motion = self.motion_data[idx]
        text = self.text_data[idx]
        length = self.length_data[idx]
        
        # Convert to torch tensors
        motion_tensor = torch.from_numpy(motion).float()
        
        # Prepare tokenized motion (for VQ-VAE)
        # Downsample by unit_length for tokenization
        tokenized_length = length // self.unit_length
        tokenized_motion = motion[::self.unit_length]
        
        return {
            "motion": motion_tensor,
            "text": text,
            "length": length,
            "tokenized_motion": torch.from_numpy(tokenized_motion).float(),
            "tokenized_length": tokenized_length,
        }
    
    def get_motion_by_text(self, text_query: str) -> Optional[Dict]:
        """Find motion by text description (for interactive use)"""
        for i, text in enumerate(self.text_data):
            if text_query.lower() in text.lower():
                return self.__getitem__(i)
        return None


class MyoSkeletonDataModule:
    """
    Data module for MyoSkeleton dataset - replaces HumanML3D datamodule
    """
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        use_simplified_joints: bool = True,
        **kwargs
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_simplified_joints = use_simplified_joints
        self.kwargs = kwargs
        
        # Create datasets
        self.train_dataset = MyoSkeletonDataset(
            data_root, split="train", use_simplified_joints=use_simplified_joints, **kwargs
        )
        self.val_dataset = MyoSkeletonDataset(
            data_root, split="val", use_simplified_joints=use_simplified_joints, **kwargs
        )
        self.test_dataset = MyoSkeletonDataset(
            data_root, split="test", use_simplified_joints=use_simplified_joints, **kwargs
        )
        
        # Set dataset properties
        self.njoints = self.train_dataset.njoints
        self.nfeats = self.train_dataset.nfeats
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for variable length sequences"""
        # Find max length in batch
        max_len = max(item["length"] for item in batch)
        
        # Pad sequences
        padded_motions = []
        texts = []
        lengths = []
        
        for item in batch:
            motion = item["motion"]
            length = item["length"]
            
            # Pad motion to max_len
            if length < max_len:
                padding = torch.zeros(max_len - length, motion.shape[1])
                padded_motion = torch.cat([motion, padding], dim=0)
            else:
                padded_motion = motion
                
            padded_motions.append(padded_motion)
            texts.append(item["text"])
            lengths.append(length)
        
        return {
            "motion": torch.stack(padded_motions),
            "text": texts,
            "length": lengths,
        }
    
    def feats2joints(self, features):
        """Convert features to joint positions for MyoSkeleton"""
        # Reshape from (batch, frames, features) to (batch, frames, joints, 3)
        batch_size, frames, _ = features.shape
        return features.reshape(batch_size, frames, self.njoints, 3) 