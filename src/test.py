# src/test.py
import argparse
from omegaconf import OmegaConf
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.image_dataset import ImageCSVDataset
from src.data.transforms import build_image_transforms
from src.models.image_backbones import build_image_backbone
from src.models.lit_module import LitClassifier

def main(args):
    cfg = OmegaConf.load(args.config)
    t = build_image_transforms(cfg.transforms.image_size, cfg.transforms.aug, is_train=False)
    data_dir = Path(cfg.paths.data_dir)
    ds = ImageCSVDataset(data_dir / cfg.dataset.test_csv, class_map_path=data_dir / cfg.dataset.class_map_path,
                         transform=t, input_col=cfg.dataset.input_col, label_col=cfg.dataset.label_col)
    dl = DataLoader(ds, batch_size=cfg.trainer.batch_size, num_workers=cfg.num_workers)

    num_classes = len(set(ds.df[cfg.dataset.label_col].unique()))
    model = build_image_backbone(cfg.model.name, num_classes, pretrained=False)
    lit = LitClassifier.load_from_checkpoint(args.ckpt, model=model)
    pl.Trainer(precision=cfg.precision).test(lit, dataloaders=dl)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/image_classification.yaml")
    ap.add_argument("--ckpt", required=True)
    main(ap.parse_args())
