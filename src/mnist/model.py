import torch.nn as nn


class MnisModel(nn.Module):
    """
    MnisModel 定义 OCR 数字识别模型
    """

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(28 * 28, 256)
        # TODO ReLU 的含义
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将 28x28 的图片展平成 784 个数字的一维数组
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x
