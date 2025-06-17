import os
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf
from typing import List


def build_callbacks(cfg: OmegaConf, logger=None, phase: str = 'train', **kwargs) -> List[Callback]:
    """
    Build a list of PyTorch Lightning callbacks based on the configuration.
    """
    callbacks = []
    
    # Add a progress bar
    callbacks.append(RichProgressBar())

    # Set up checkpointing if in training phase
    if phase == 'train':
        # Base checkpointing config
        checkpoint_dir = os.path.join(cfg.FOLDER_EXP, "checkpoints")
        
        # 1. Save the last checkpoint
        callbacks.append(ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='last-{epoch}',
            save_last=True,
            every_n_epochs=1,
        ))
        
        # 2. Save checkpoints based on a monitored metric (e.g., validation loss)
        if cfg.LOGGER.get("MONITOR_METRIC"):
            callbacks.append(ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='best-{' + cfg.LOGGER.MONITOR_METRIC + ':.4f}-{epoch}',
                monitor=cfg.LOGGER.MONITOR_METRIC,
                mode=cfg.LOGGER.get("MONITOR_MODE", "min"),
                save_top_k=cfg.LOGGER.get("SAVE_TOP_K", 3),
            ))
            
        # 3. Save checkpoints periodically
        if cfg.LOGGER.get("SAVE_PERIOD"):
            callbacks.append(ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='periodic-{epoch}',
                every_n_epochs=cfg.LOGGER.SAVE_PERIOD,
                save_on_train_epoch_end=True
            ))

    # Add a learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    return callbacks

class progressBar(RichProgressBar):
    def __init__(self, ):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

class progressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1):
        # Metric to monitor
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        self.logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            self.logger.info("Sanity checking ok.")

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding

        if trainer.current_epoch % self.log_every_n_steps == 0:
            metrics_str = []

            losses_dict = trainer.callback_metrics
            for metric_name, dico_name in self.metric_monitor.items():
                if dico_name in losses_dict:
                    metric = losses_dict[dico_name].item()
                    metric = metric_format.format(metric)
                    metric = f"{metric_name} {metric}"
                    metrics_str.append(metric)

            line = line + ": " + "   ".join(metrics_str)

        self.logger.info(line)
