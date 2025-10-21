import argparse, json
from pathlib import Path
import torch
from PIL import Image
from src.data.transforms import build_image_transforms
from src.models.image_backbones import build_image_backbone
from src.models.lit_module import LitClassifier

@torch.no_grad()
def main(args):
    cfg_path = Path(args.config)
    import yaml; cfg = yaml.safe_load(cfg_path.read_text())
    t = build_image_transforms(cfg["transforms"]["image_size"], cfg["transforms"]["aug"], is_train=False)

    class_map = json.loads(Path(cfg["paths"]["data_dir"]) .joinpath(cfg["dataset"]["class_map_path"]).read_text())
    idx_to_class = {int(v): k for k, v in class_map.items()}
    num_classes = len(idx_to_class)

    model = build_image_backbone(cfg["model"]["name"], num_classes, pretrained=False)
    lit = LitClassifier.load_from_checkpoint(args.ckpt, model=model)
    lit.eval()

    img = Image.open(args.image).convert("RGB")
    x = t(img).unsqueeze(0)
    logits = lit(x)
    pred = logits.argmax(dim=1).item()
    print("Prediction:", idx_to_class.get(pred, pred))

    if args.export_onnx:
        torch.onnx.export(lit.model, x, args.export_onnx, input_names=["input"], output_names=["logits"], opset_version=17)
        print("Exported ONNX:", args.export_onnx)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/image_classification.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--export_onnx")
    main(ap.parse_args())
