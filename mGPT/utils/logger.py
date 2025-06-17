from pathlib import Path
import os
import time
import logging
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from typing import Optional

@rank_zero_only
def setup_logger(log_dir: str, phase: str = 'train') -> logging.Logger:
    """
    Set up a logger for training and evaluation.

    Args:
        log_dir (str): The directory to save the log file.
        phase (str): The phase of the experiment ('train', 'test', 'demo', etc.).

    Returns:
        logging.Logger: The configured logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = log_dir / f'{phase}_{time_str}.log'

    # Set up the logger
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Add a stream handler to also print to console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(head))
    logger.addHandler(console)

    logger.info(f"Logger set up. Log file: {log_file}")
    return logger

@rank_zero_only
def setup_experiment_dir(cfg: OmegaConf, phase: str = 'train') -> Path:
    """
    Set up the experiment directory and save the configuration file.

    Args:
        cfg (OmegaConf): The configuration object.
        phase (str): The phase of the experiment.

    Returns:
        Path: The path to the experiment directory.
    """
    root_dir = Path(cfg.FOLDER)
    
    # Create a unique experiment name
    model_name = cfg.model.target.split('.')[-2]
    cfg_name = Path(cfg.NAME).stem
    exp_name = f"{model_name}_{cfg_name}"
    
    # Add a timestamp to avoid overwriting
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    
    exp_dir = root_dir / f"{exp_name}_{time_str}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the config to the experiment directory
    config_path = exp_dir / f'config_{phase}.yaml'
    OmegaConf.save(config=cfg, f=config_path)
    
    # Store the experiment directory path in the config for easy access
    cfg.FOLDER_EXP = str(exp_dir)
    
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir
