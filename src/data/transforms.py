import torchvision.transforms as T

def build_image_transforms(image_size=224, aug=None, is_train=True):
    aug = aug or {}
    tfm = []
    if is_train:
        if aug.get("random_crop", True):
            tfm.append(T.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
        else:
            tfm += [T.Resize(int(image_size*1.15)), T.CenterCrop(image_size)]
        if aug.get("random_flip", True):
            tfm.append(T.RandomHorizontalFlip())
        if aug.get("color_jitter", False):
            tfm.append(T.ColorJitter(0.2,0.2,0.2,0.1))
    else:
        tfm += [T.Resize(int(image_size*1.15)), T.CenterCrop(image_size)]

    tfm += [T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
    return T.Compose(tfm)
