import torch
from torch import nn

from mnist.model import MnisModel

from ..utils.device import DEVICE

# 设置随机种子：保证每次运行结果一致
torch.manual_seed(1024)


def start(model: MnisModel, train_loader: torch.utils.data.DataLoader):
    # TODO CrossEntropyLoss、optimizer 的含义
    criterion = nn.CrossEntropyLoss()  # 定义损失函数（交叉熵损失）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 定义优化器（随机梯度下降）

    epochs = 5  # 设置训练轮数为 5 轮

    for epoch in range(epochs):  # 循环训练 5 轮
        running_loss = 0.0  # 初始化累计损失

        for i, (images, labels) in enumerate(train_loader):
            # 将数据迁移到与模型相同的设备（GPU/CPU）
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 清除上一次迭代的梯度，确保每次迭代使用的是当前批次的梯度
            optimizer.zero_grad()

            # 将图片输入模型得到预测
            outputs = model(images)

            # 计算预测与真实标签的损失
            loss = criterion(outputs, labels)
            # 计算梯度
            loss.backward()
            # 使用梯度更新参数
            optimizer.step()
            # 累加损失值
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, 每次训练损失: {loss.item():.4f} 总损失: {running_loss / len(train_loader):.4f}")
