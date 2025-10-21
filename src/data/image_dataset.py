from pathlib import Path
import json, pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImageCSVDataset(Dataset):
    def __init__(self, csv_path, class_map_path=None, transform=None, input_col="path", label_col="label"):
        self.df = pd.read_csv(csv_path)
        self.input_col, self.label_col = input_col, label_col
        self.transform = transform
        self.class_map = None
        if class_map_path and Path(class_map_path).exists():
            self.class_map = json.loads(Path(class_map_path).read_text(encoding="utf-8"))
        # 基础校验
        assert self.input_col in self.df.columns and self.label_col in self.df.columns
        # 过滤缺文件的行
        self.df = self.df[self.df[self.input_col].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.input_col]).convert("RGB")
        if self.transform: img = self.transform(img)
        label = int(row[self.label_col])
        return img, label
