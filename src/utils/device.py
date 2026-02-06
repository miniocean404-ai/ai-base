# 自动检测可用设备：优先使用 GPU（CUDA），其次 CPU
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
