import matplotlib.pyplot as plt
import torch
import torchvision

from .model import MnisModel
from ..utils.device import DEVICE


def save_model(model, path):
    # debugger 时候可以查看模型权重，可以打印参数: model.fc1.weight、model.fc2.weight、model.fc3.weight
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")


def load_model(path):
    model = MnisModel()
    # map_location 确保模型权重加载到当前可用设备，避免在无 GPU 机器上加载 GPU 模型时报错
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    return model


def get_mnist_real_data(dataset: torchvision.datasets.MNIST):
    """
    获取 MNIST 数据集的标签和像素, 并显示图片
    """
    # 打印第一个数据的具体图片肉眼看的数值，输出 tensor(7)
    print(dataset.targets[0])

    # 打印图片的原始像素, 输出为一个 28 * 28 的 8(0-255) 位整数张量
    print(dataset.data[0])

    plt.imshow(dataset.data[0], cmap="gray")
    plt.show()


# pass 用于留空, 防止语法报错, pass 也可以替换为 ..., 但是 ... 多用于 pyi 文件
def test():
    pass
