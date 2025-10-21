import torch
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
def setup_cudnn(deterministic: bool, benchmark: bool):
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = bool(deterministic)
    cudnn.benchmark = bool(benchmark and not deterministic)
