import importlib
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf
from os.path import join as pjoin
import os
from typing import List, Any, Type

def load_config(cfg_path: str, cli_args: List[str] = []) -> OmegaConf:
    """
    Load a configuration from a YAML file and merge it with command-line arguments.
    """
    cfg = OmegaConf.load(cfg_path)
    # Merge with CLI args
    cli_cfg = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg

def get_obj_from_str(string: str, reload: bool = False) -> Type[Any]:
    """
    Get an object (class or function) from a string representation.
    e.g., "mGPT.models.mgpt.MotionGPT" -> <class 'mGPT.models.mgpt.MotionGPT'>
    """
    module_name, cls_name = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module_name)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module_name, package=None), cls_name)


def instantiate_from_config(config: OmegaConf) -> Any:
    """
    Instantiate an object from an OmegaConf configuration.
    Requires the config to have a "target" key with the object's path.
    """
    if "target" not in config:
        raise KeyError("Config must have a 'target' key to specify the object to instantiate.")
    
    params = config.get("params", {}) or {}
    return get_obj_from_str(config["target"])(**params)


def resume_from_checkpoint(cfg: OmegaConf) -> OmegaConf:
    """
    Update a configuration to resume training from a checkpoint.
    This function is designed to be used with a specific directory structure
    and may need adjustment for different logging/checkpointing setups.
    """
    if not cfg.TRAIN.get("RESUME"):
        return cfg

    resume_path = cfg.TRAIN.RESUME
    if not os.path.isdir(resume_path):
        raise ValueError(f"Resume path '{resume_path}' is not a valid directory.")

    # Find the last checkpoint
    checkpoint_path = pjoin(resume_path, "checkpoints", "last.ckpt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 'last.ckpt' not found in '{pjoin(resume_path, 'checkpoints')}'")
    
    cfg.TRAIN.PRETRAINED = checkpoint_path
    
    # Find the wandb run ID to resume logging
    wandb_dir = pjoin(resume_path, "wandb", "latest-run")
    if os.path.isdir(wandb_dir):
        wandb_files = os.listdir(wandb_dir)
        try:
            wandb_run_file = next(f for f in wandb_files if f.startswith("run-") and f.endswith(".wandb"))
            wandb_id = wandb_run_file.replace("run-", "").replace(".wandb", "")
            
            # Update the config to resume the wandb run
            if not cfg.get("LOGGER"):
                cfg.LOGGER = {}
            if not cfg.LOGGER.get("WANDB"):
                cfg.LOGGER.WANDB = {"params": {}}
                
            cfg.LOGGER.WANDB.params.id = wandb_id
            cfg.LOGGER.WANDB.params.resume = "allow"
            
        except StopIteration:
            print(f"Warning: Could not find a wandb run file in '{wandb_dir}'. A new run will be started.")

    return cfg


def parse_args(phase: str = "train") -> OmegaConf:
    """
    Parse command-line arguments and load the corresponding configuration.
    This function is now simplified and delegates most of the work to the
    training/demo scripts themselves.
    """
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to the configuration file.")
    
    # Allow unknown arguments to be parsed by OmegaConf
    args, unknown = parser.parse_known_args()
    
    # Load the base configuration file
    cfg = OmegaConf.load(args.cfg)
    
    # Merge with any command-line overrides
    cli_cfg = OmegaConf.from_cli(unknown)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    # Set up debug mode if specified
    if cfg.get("DEBUG"):
        cfg.NAME = "debug-" + cfg.get("NAME", "default")
        if cfg.get("LOGGER") and cfg.LOGGER.get("WANDB"):
            cfg.LOGGER.WANDB.params.offline = True
        if cfg.get("LOGGER"):
            cfg.LOGGER.VAL_EVERY_STEPS = 1
            
    # Handle resuming from a checkpoint
    if phase == "train" and cfg.TRAIN.get("RESUME"):
        cfg = resume_from_checkpoint(cfg)
        
    return cfg

# A simple function to load a config, kept for compatibility with older scripts
def load_config_simple(cfg_path: str):
    return OmegaConf.load(cfg_path)
