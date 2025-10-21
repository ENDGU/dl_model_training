import argparse, json
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.utils.seed import set_seed, setup_cudnn
from src.data.transforms import build_image_transforms
from src.data.image_dataset import ImageCSVDataset
from src.models.image_backbones import build_image_backbone
from src.models.lit_module import LitClassifier
from src.callbacks import build_callbacks

def load_cfg(cfg_path):
    base = OmegaConf.load("configs/default.yaml")
    user = OmegaConf.load(cfg_path) if cfg_path else OmegaConf.create()
    return OmegaConf.merge(base, user)

def build_dataloaders(cfg, class_map_path):
    t_train = build_image_transforms(cfg.transforms.image_size, cfg.transforms.aug, True)
    t_val   = build_image_transforms(cfg.transforms.image_size, cfg.transforms.aug, False)
    data_dir = Path(cfg.paths.data_dir)
    train_ds = ImageCSVDataset(data_dir / cfg.dataset.train_csv, class_map_path=class_map_path, transform=t_train,
                               input_col=cfg.dataset.input_col, label_col=cfg.dataset.label_col)
    val_ds   = ImageCSVDataset(data_dir / cfg.dataset.val_csv,   class_map_path=class_map_path, transform=t_val,
                               input_col=cfg.dataset.input_col, label_col=cfg.dataset.label_col)
    test_ds  = ImageCSVDataset(data_dir / cfg.dataset.test_csv,  class_map_path=class_map_path, transform=t_val,
                               input_col=cfg.dataset.input_col, label_col=cfg.dataset.label_col)
    dl_args = dict(batch_size=cfg.trainer.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return (DataLoader(train_ds, shuffle=True,  **dl_args),
            DataLoader(val_ds,   shuffle=False, **dl_args),
            DataLoader(test_ds,  shuffle=False, **dl_args),
            len(set(train_ds.df[cfg.dataset.label_col].unique())))

def main(args):
    cfg = load_cfg(args.config)
    set_seed(cfg.seed, cfg.deterministic)
    setup_cudnn(cfg.deterministic, cfg.cudnn_benchmark)

    outputs = Path(cfg.paths.outputs_dir); outputs.mkdir(parents=True, exist_ok=True)
    class_map_path = Path(cfg.paths.data_dir) / cfg.dataset.class_map_path
    # 若不存在，训练时会用到类别数，所以这里兜底从train.csv推断
    if not class_map_path.exists():
        from collections import OrderedDict
        import pandas as pd
        df = pd.read_csv(Path(cfg.paths.data_dir)/cfg.dataset.train_csv)
        labels = sorted(df[cfg.dataset.label_col].unique().tolist())
        json.dump(OrderedDict((str(i),int(i)) for i in labels),
                  open(class_map_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    train_dl, val_dl, test_dl, num_classes = build_dataloaders(cfg, class_map_path)
    model = build_image_backbone(cfg.model.name, num_classes, pretrained=cfg.model.pretrained)
    lit = LitClassifier(model, num_classes, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay,
                        scheduler=cfg.scheduler.name, warmup_epochs=cfg.scheduler.warmup_epochs)

    cbs = build_callbacks(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        devices=cfg.trainer.devices, accelerator=cfg.trainer.accelerator,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=cbs,
        default_root_dir=cfg.paths.outputs_dir
    )
    trainer.fit(lit, train_dl, val_dl)
    print("Best checkpoint:", trainer.checkpoint_callback.best_model_path)

    # 训练后在测试集上评估一次（确保只用最优权重）
    if trainer.checkpoint_callback.best_model_path:
        lit = LitClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=model)
    trainer.test(lit, dataloaders=test_dl)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", default="configs/image_classification.yaml")
    main(ap.parse_args())
