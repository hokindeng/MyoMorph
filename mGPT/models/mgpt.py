import numpy as np
import os
import random
import torch
import time
from mGPT.config import instantiate_from_config
from os.path import join as pjoin
from mGPT.losses.mgpt import GPTLosses
from mGPT.models.base import BaseModel
from .base import BaseModel
import json
import mGPT.render.matplot.plot_3d_global as plot_3d


class MotionGPT(BaseModel):
    """
    MotionGPT: A Unified Motion-Language Model.

    This model is designed to handle multiple stages of training:
    1. VQ-VAE: Training the motion tokenizer.
    2. LM Pre-training: Pre-training the language model on motion and text data.
    3. LM Instruction Tuning: Fine-tuning the language model on specific tasks.
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'], logger=False)
        self.datamodule = datamodule
        
        # Instantiate the motion VAE (tokenizer)
        self.vae = instantiate_from_config(self.hparams.cfg.model.params.motion_vae)
        
        # Instantiate the language model
        self.lm = instantiate_from_config(self.hparams.cfg.model.params.lm)

        # Freeze layers based on the training stage
        self.freeze_layers()

        # Instantiate the loss functions for each split
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.cfg.model.params.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Keep a direct reference to the feature-to-joint conversion function
        self.feats2joints = self.datamodule.feats2joints

    def freeze_layers(self):
        """Freeze parts of the model based on the training stage."""
        stage = self.hparams.cfg.model.params.stage
        if 'lm' in stage:
            # If we are training the language model, freeze the VAE
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False
        elif 'vae' in stage:
            # If we are training the VAE, freeze the LM
            self.lm.training = False
            for p in self.lm.parameters():
                p.requires_grad = False

    def _decode_motion(self, outputs, task, batch=None):
        """Helper function to decode motion tokens into features."""
        feats_rst_lst = []
        lengths = []
        max_len = 0

        for i in range(len(outputs)):
            if task == "pred":
                motion = self.vae.decode(torch.cat((batch["motion"][i], outputs[i])))
            else:  # t2m, m2t, inbetween
                motion = self.vae.decode(outputs[i])
                lengths.append(motion.shape[1])

            if motion.shape[1] > max_len:
                max_len = motion.shape[1]

            if task == "inbetween":
                lengths_ref = batch["length"]
                motion = torch.cat(
                    (batch["motion_heading"][i][None],
                     motion[:, lengths_ref[i] // 4:lengths_ref[i] // 4 * 3, :],
                     batch["motion_tailing"][i][None]),
                    dim=1
                )
            feats_rst_lst.append(motion)

        # Pad and concatenate features
        feats_rst = torch.zeros((len(feats_rst_lst), max_len, feats_rst_lst[0].shape[-1]), device=self.device)
        for i, motion in enumerate(feats_rst_lst):
            feats_rst[i, :motion.shape[1], :] = motion
            
        return feats_rst, lengths

    def forward(self, batch, task="t2m"):
        """Main forward pass for inference."""
        texts = batch["text"]
        
        # Generate sequences from the language model
        outputs, output_texts = self.lm.generate_direct(texts, do_sample=True)

        # Decode the generated tokens into motion features
        feats_rst, lengths = self._decode_motion(outputs, task, batch)
        
        # Convert features to joint positions
        joints_rst = self.feats2joints(feats_rst)

        return {
            "texts": output_texts,
            "feats": feats_rst,
            "joints": joints_rst,
            "length": lengths
        }
    def train_lm_forward(self, batch):
        """Forward pass for training the language model."""
        tokens_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]
        
        if self.hparams.cfg.model.params.condition == 'caption':
            texts = [random.choice(captions) for captions in batch['all_captions']]

        # Language model forward pass
        outputs = self.lm(texts, tokens_ref, lengths, tasks)
        return {'outputs': outputs}

    def train_vae_forward(self, batch):
        """Forward pass for training the VQ-VAE."""
        feats_ref = batch["motion"]
        joints_ref = self.feats2joints(feats_ref)
        
        # VAE forward pass
        feats_rst, loss_commit, perplexity = self.vae(feats_ref)
        joints_rst = self.feats2joints(feats_rst)
        
        return {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "loss_commit": loss_commit,
            "perplexity": perplexity,
        }
    
    def validation_step(self, batch, batch_idx):
        """Centralized validation logic."""
        task = self.hparams.cfg.model.params.task
        stage = self.hparams.cfg.model.params.stage
        
        rs_set = self._get_validation_output(batch, task, stage)
        if rs_set:
            self.compute_and_log_metrics(rs_set, batch)

    def _get_validation_output(self, batch, task, stage):
        """Helper to get the output for a validation step based on task and stage."""
        if stage == "vae":
            return self.val_vae_forward(batch)
        elif "lm" in stage:
            if task == "t2m":
                return self.val_t2m_forward(batch)
            elif task == "m2t":
                return self.val_m2t_forward(batch)
            elif task in ["m2m", "pred", "inbetween"]:
                return self.val_m2m_forward(batch, task)
        return None

    def test_step(self, batch, batch_idx):
        task = self.hparams.cfg.model.params.task
        stage = self.hparams.cfg.model.params.stage

        rs_set = self._get_validation_output(batch, task, stage)
        if rs_set:
            self.compute_and_log_metrics(rs_set, batch)
            
            # Return specific outputs for testing
            if task == "t2m":
                return rs_set["joints_rst"], rs_set["length"], rs_set["joints_ref"]
            elif task == "m2t":
                return rs_set["t_pred"], batch["length"]
        
        return None
    
    def allsplit_step(self, split: str, batch, batch_idx):
        if split == 'train':
            return self.training_step(batch, batch_idx)
        elif split == 'val':
            return self.validation_step(batch, batch_idx)
        elif split == 'test':
            return self.test_step(batch, batch_idx)

    def compute_and_log_metrics(self, rs_set, batch):
        """Compute and log all relevant metrics for a given validation output."""
        task = self.hparams.cfg.model.params.task
        
        # Define the metrics to compute based on the task
        if task == "t2m":
            metrics_to_compute = ['TM2TMetrics', 'TemosMetric', 'MRMetrics']
        elif task == "m2t":
            metrics_to_compute = ['M2TMetrics']
        elif task in ["m2m", "pred"]:
            metrics_to_compute = ['PredMetrics']
        else: # "inbetween" and "vae"
            metrics_to_compute = []

        if self.trainer.datamodule.is_mm:
            metrics_to_compute.append('MMMetrics')

        for metric_name in metrics_to_compute:
            if hasattr(self.metrics, metric_name):
                metric_calculator = getattr(self.metrics, metric_name)
                # This part needs to be adapted based on the specific inputs of each metric
                # For now, we assume a generic update, but this will need refinement
                # Example for TM2TMetrics
                if metric_name == "TM2TMetrics":
                    metric_calculator.update(
                        feats_ref=rs_set["m_ref"],
                        feats_rst=rs_set["m_rst"],
                        lengths_ref=batch["length"],
                        lengths_rst=rs_set['length'],
                        word_embs=batch.get("word_embs"),
                        pos_ohot=batch.get("pos_ohot"),
                        text_lengths=batch.get("text_len"),
                    )
                # Add other metric updates here...

    def training_step(self, batch, batch_idx):
        stage = self.hparams.cfg.model.params.stage
        if "lm" in stage:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_train'].update(rs_set)
        elif "vae" in stage:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_train'].update(rs_set)
        else:
            loss = None
        return loss
