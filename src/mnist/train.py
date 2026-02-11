import torch
from torch import nn

from mnist.model import MnisModel

from ..utils.device import DEVICE

# 设置随机种子：保证每次运行结果一致
torch.manual_seed(1024)


def start(model: MnisModel, train_loader: torch.utils.data.DataLoader):
    criterion = nn.CrossEntropyLoss()  # 定义损失函数（交叉熵损失算法）: loss = -ln(p），内部已经包含了 softmax 操作
    # 定义优化器（随机梯度下降）
    # 参数 1: 调整哪些参数
    # 参数 2：初始化的学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 5  # 设置训练轮数为 5 轮

    for epoch in range(epochs):  # 循环训练 5 轮
        running_loss = 0.0  # 初始化累计损失

        for i, (images, labels) in enumerate(train_loader):
            # 将数据迁移到与模型相同的设备（GPU/CPU）
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 清除上一次迭代的梯度，确保每次迭代使用的是当前批次的梯度,
            # 也可以使用 model.zero_grad() 代替, 但是需要手动写学习率调整代码，需要添加 with torch.no_grad()
            # with torch.no_grad(): 不计算梯度
            # 训练时: 默认情况下所有 pytorch 的计算，包括更新参数的计算都会影响下一次 loss.backward() 梯度的计算结果,
            # 但是更新参数本身并不是模型的一部分，所以使用 with torch.no_grad() 通知 pytorch 将这些计算排除在梯度计算之外
            optimizer.zero_grad()

            # 将图片输入模型得到预测
            outputs = model(images)

            # 计算预测与真实标签的损失
            loss = criterion(outputs, labels)
            # 反向传播: 沿着神经网络的结构，从输出层反向向输入层推导。利用微积分中的链式法则，计算出损失函数对模型中每一个参数（权重 w 和偏置 b）的梯度（Gradient, 也就是求导斜率）。
            loss.backward()
            # 使用 backward 计算的梯度更新参数
            optimizer.step()
            # 累加损失值
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, 每次训练损失: {loss.item():.4f} 总损失: {running_loss / len(train_loader):.4f}")
