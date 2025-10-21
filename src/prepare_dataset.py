# src/prepare_dataset.py
import argparse, json, random, shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main(args):
    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 示例：从目录结构 raw/<class>/*.jpg 生成清单
    rows = []
    classes = sorted([p.name for p in raw.iterdir() if p.is_dir()])
    assert classes, f"No class folders found in {raw}"
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for c in classes:
        for img in (raw / c).rglob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                rows.append({"path": str(img.resolve()), "label": class_to_idx[c]})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # 清理缺失文件
    ok = df["path"].apply(lambda p: Path(p).exists())
    df = df[ok].reset_index(drop=True)

    # stratified split
    train_df, tmp = train_test_split(
        df, test_size=args.val_ratio + args.test_ratio,
        stratify=df["label"], random_state=args.seed
    )
    rel = args.test_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        tmp, test_size=rel, stratify=tmp["label"], random_state=args.seed
    )

    # 保存到 processed/ 下，路径保持绝对路径以避免工作目录变化带来的坑
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    with open(out / "class_index.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    print(f"Prepared: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
    print(f"Classes: {class_to_idx}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
