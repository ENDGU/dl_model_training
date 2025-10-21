import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pathlib import Path

def build_callbacks(cfg):
    ckpt_dir = Path(cfg.paths.outputs_dir) / cfg.paths.ckpt_dirname
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}-{val_acc:.4f}",
        monitor=cfg.callbacks.model_checkpoint.monitor,
        mode=cfg.callbacks.model_checkpoint.mode,
        save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
        save_last=cfg.callbacks.model_checkpoint.save_last
    )
    es = EarlyStopping(
        monitor=cfg.callbacks.early_stopping.monitor,
        mode=cfg.callbacks.early_stopping.mode,
        patience=cfg.callbacks.early_stopping.patience
    )
    lrm = LearningRateMonitor(logging_interval="epoch")
    return [ckpt, es, lrm]
