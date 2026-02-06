# 训练图像识别神经网络模型
import os
import random  # 导入随机数库，用于随机选择图片
from os import path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn  # 导入神经网络模块
import torchvision  # 导入计算机视觉库
import torchvision.transforms as transforms  # 导入图像变换工具


def load_data(data_path: str):
    """
    数据准备，下载的数据本质上是 28 * 28 的灰度图片
    """
    # 定义图像预处理步骤
    transform = transforms.Compose(
        [
            # 将图片转换为张量（数字矩阵）
            torchvision.transforms.ToTensor(),
            # 将像素值标准化到 - 1 到 1 之间
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # 下载并加载 MNIST 训练数据集（手写数字 0-9）
    train_dataset = torchvision.datasets.MNIST(
        root=data_path,  # 数据保存路径
        train=True,  # 加载训练集
        transform=transform,  # 应用上面定义的预处理
        download=True,  # 如果数据不存在就下载
    )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,  # 使用上面的训练数据集
        batch_size=64,  # 每次训练使用 64 张图片
        shuffle=True,  # 随机打乱数据顺序
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_path,  # 数据保存路径
        train=False,  # 加载测试集
        transform=transform,  # 应用预处理
        download=True,  # 如果数据不存在就下载
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,  # 使用测试数据集
        batch_size=64,  # 每次处理 64 张图片
        shuffle=False,  # 不打乱测试数据顺序
    )

    return train_loader, test_loader, test_dataset


class MnistModel(nn.Module):
    """
    MnistModel 定义 OCR 数字识别模型，如果要使用模型，那么加载模型时候也必须创建这个模型的实例，并填入模型权重数据
    """

    def __init__(self) -> None:
        super().__init__()

        # Linear 层用于将指定维度的张量通过线性变换转化为输出层的维度的张量
        self.layer1 = nn.Linear(28 * 28, 256)
        # ReLU 激活：把所有负值变成 0，保留正值
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        # 将 28 x 28 的图片展平成 784 个数字的一维数组，-1 代表自动计算批次大小，但是 -1 设置不能过多否则报错
        x = x.view(-1, 28 * 28)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x


def verify_ui(model: MnistModel, data_set: torchvision.datasets.MNIST):
    # 将模型设置为评估模式（不训练）
    model.eval()

    # 创建 1 行 5 列的 ui 图表
    # fig 代表整个图形对象（Figure），你可以用它来设置全局属性，比如标题、背景色、保存图像等
    # axes 是一个包含 5 个 Axes 对象的数组（因为是 1×5 布局），每个 Axes 对应一个子图，用于绘制具体内容（如图像、曲线等）。
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))

    for i in range(5):
        idx = random.randint(0, len(data_set) - 1)
        # 获取图片和真实值
        image, label = data_set[idx]

        print(image.size())
        print(image.unsqueeze(0).size())

        # 不计算梯度（节省内存）, with 语法会在执行完后自动执行 __exit__ 方法释放资源
        with torch.no_grad():
            # 将图片输入模型获得预测，
            # unsqueeze 表示在第 0 维（即最前面）增加一个维度, 之前：[通道数, 高度，宽度] 之后：[批次大小，通道数，高度，宽度]
            # unsqueeze(dim) dim 是增加张量维度的索引(张量.size() 的值的索引)、-1 为最后面添加维度
            output = model(image.unsqueeze(0))

            # softmax 作用是将所有数值归一化为概率值，并且概率值之和为 1
            output = torch.softmax(output, dim=1)

            # 返回指定维度上最大值的索引。
            # 几维数组就是几维张量
            # dim=1 的含义为在张量为 2 的维度上是 tensor[0][0] 获取到的值来进行操作, dim=0 的含义为在张量为 2 的维度上是 tensor[0] 获取到的值来进行操作
            pred = torch.argmax(output, dim=1).item()

        axes[i].imshow(image.squeeze(), cmap="gray")  # 显示 UI 灰度图片
        axes[i].set_title(f"Predicted: {pred}")  # 设置 UI 为预测结果
        axes[i].axis("off")  # 隐藏坐标轴

    # 显示 UI 图表
    plt.show()


def train_model(model: MnistModel, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader):
    """
    训练模型
    """
    model.train()
    criterion = nn.CrossEntropyLoss()  # 定义损失函数（交叉熵损失）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 定义优化器（随机梯度下降）

    epochs = 5  # 设置训练轮数为 5 轮
    for epoch in range(epochs):  # 循环训练 5 轮
        running_loss = 0.0  # 初始化累计损失
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # 清除上一次迭代的梯度，确保每次迭代使用的是当前批次的梯度
            outputs = model(images)  # 将图片输入模型得到预测
            loss = criterion(outputs, labels)  # 计算预测与真实标签的损失
            loss.backward()
            optimizer.step()  # 使用梯度更新参数
            running_loss += loss.item()  # 累加损失值

        # Epoch 代表训练第几轮数据
        print(
            f"Epoch {epoch + 1}/{epochs}, 当前损失: {loss.item():.4f}, 当前 {epoch + 1} 轮总平均损失值: {running_loss / len(train_loader):.4f}"
        )


def save_model(model: MnistModel, model_path: str):
    if not path.exists(path.dirname(model_path)):
        os.makedirs(path.dirname(model_path))

    # debugger 时候可以查看模型权重，可以打印参数: model.fc1.weight、model.fc2.weight、model.fc3.weight
    torch.save(model.state_dict(), model_path)


data_path = "data/mnist"
model_path = "model/mnist.bin"
train_loader, test_loader, test_dataset = load_data(data_path)
mnist_model = MnistModel()


def train_main():
    # 没有训练并且没有加载模型的时候测试一下模型准确率
    verify_ui(mnist_model, test_dataset)

    # 训练模型
    train_model(mnist_model, train_loader, test_loader)
    # 训练后测试一下模型准确率
    verify_ui(mnist_model, test_dataset)
    # 训练后保存模型
    save_model(mnist_model, model_path)


def use_main():
    mnist_model.load_state_dict(torch.load(model_path))
    print("✅ 本地加载模型成功")
    mnist_model.eval()
    verify_ui(mnist_model, test_dataset)


def main():
    use_main()
    # train_main()


if __name__ == "__main__":
    main()
