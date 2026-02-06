import random
from typing import cast

import matplotlib.pyplot as plt
import torch
import torchvision

from ..utils.device import DEVICE


def start(model, data_set: torchvision.datasets.MNIST):
    """
    start 预测模型, predict(预测)

    Args:
        model (MnisModel): 模型
        data_set (torchvision.datasets.MNIST): 数据集
    """
    # 将模型设置为评估模式（也就是大模型运行时, 不训练）
    model.eval()

    # 创建 1 行 5 列的 ui 图表
    # fig 代表整个图形对象（Figure），你可以用它来设置全局属性，比如标题、背景色、保存图像等
    # axes 是一个包含 5 个 Axes 对象的数组（因为是 1×5 布局），每个 Axes 对应一个子图，用于绘制具体内容（如图像、曲线等）。
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))

    for i in range(5):
        random_idx = random.randint(0, len(data_set) - 1)

        # 获取图片和真实标签
        image, label = cast(tuple[torch.Tensor, int], data_set[random_idx])

        # 打印图片和图片的维度
        print("image.size()", image.size())
        print("image.unsqueeze(0).size()", image.unsqueeze(0).size())

        # 不计算梯度（节省内存）, with 语法会在执行完后自动执行 __exit__ 方法释放资源
        with torch.no_grad():
            # 将图片输入模型获得预测,
            # unsqueeze 用于将二维数组图数据 [[1, 2], [3, 4]] 片转化为四维数组 [[[1, 2], [3, 4]]], 之前：[高度, 宽度, 通道数] -> [224, 224, 3] (这是一张图片)，之后：[批次大小, 高度, 宽度, 通道数] -> [1, 224, 224, 3] (这是一个包含一张图片的批次)
            # 参数 dim=0 参数代表在哪个梯度进行扩展, 在最前面加维度、dim=1: 在原来的第 0 维和第 1 维之间加维度、dim=-1: 在最后面加维度（非常常用）
            output = model(image.unsqueeze(0).to(DEVICE))

            # softmax 作用是将所有数值归一化为概率值，并且概率值之和为 1
            output = torch.softmax(output, dim=1)

            # 返回指定维度上最大值的索引。
            # 几维数组就是几维张量
            # dim=1 的含义为在张量为 2 的维度上是 tensor[0][0] 获取到的值来进行操作, dim=0 的含义为在张量为 2 的维度上是 tensor[0] 获取到的值来进行操作
            pred = torch.argmax(output, dim=1).item()  # 找到概率最大的类别的索引的值

        axes[i].imshow(image.squeeze(), cmap="gray")  # 显示 UI 灰度图片
        axes[i].set_title(f"Predicted: {pred}")  # 设置 UI 标题显示预测结果
        axes[i].axis("off")  # 隐藏坐标轴

    # 显示 UI 图表
    plt.show()
