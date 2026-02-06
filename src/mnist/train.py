import random

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

# 设置随机种子：保证每次运行结果一致
torch.manual_seed(1024)

# 自动检测可用设备：优先使用 GPU（CUDA），其次 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def download_mnist():
    # 定义图像预处理步骤
    transform = transforms.Compose(
        [
            # 将图片转换为张量（数字矩阵）
            torchvision.transforms.ToTensor(),
            # 将像素值标准化到 - 1 到 1 之间
            # 因为图片的像素值范围是 0-255, 数值太大了, 线性变换的过程就是不断的乘除加减, 如果输入非常大, 那么结果也会很大
            # 会产生两个问题: 模型难以收敛（学不会）、可能数值溢出直接导致模型崩溃，并且现在的 gpu h100，为了追求效率往往使用 8 位或者 4 位浮点数来进行计算，这写低精度的格式在处理大数的时候误差会非常大
            # 业界标准做法是将输入的数字都转为比较小的小数, (转化方法没有限制，可以写为自己的)
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # 下载并加载 MNIST 训练数据集（手写数字 0-9）, train_dataset 形状为: torch.Size([60000, 28, 28])
    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist",  # 数据保存路径
        train=True,  # 加载训练集
        transform=transform,  # 应用上面定义的预处理
        download=True,  # 如果数据不存在就下载
    )

    # 创建训练数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,  # 使用上面的训练数据集
        batch_size=64,  # 每次训练使用 64 张图片
        shuffle=True,  # 随机打乱数据顺序
    )

    # 下载并加载 MNIST 测试数据集
    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist",  # 数据保存路径
        train=False,  # 加载测试集
        transform=transform,  # 应用预处理
        download=True,  # 如果数据不存在就下载
    )

    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,  # 使用测试数据集
        batch_size=64,  # 每次处理 64 张图片
        shuffle=False,  # 不打乱测试数据顺序
    )

    return train_loader, test_loader, train_dataset, test_dataset


class MnisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 256)
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


def train(model, train_loader):
    criterion = nn.CrossEntropyLoss()  # 定义损失函数（交叉熵损失）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 定义优化器（随机梯度下降）

    epochs = 5  # 设置训练轮数为 5 轮

    for epoch in range(epochs):  # 循环训练 5 轮
        running_loss = 0.0  # 初始化累计损失

        for i, (images, labels) in enumerate(train_loader):
            # 将数据迁移到与模型相同的设备（GPU/CPU）
            images, labels = images.to(device), labels.to(device)

            # 清除上一次迭代的梯度，确保每次迭代使用的是当前批次的梯度
            optimizer.zero_grad()

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


def predict(model, data_set: torchvision.datasets.MNIST):
    model.eval()  # 将模型设置为评估模式（不训练）
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))  # 创建 1 行 5 列的 ui 图表

    for i in range(5):
        idx = random.randint(0, len(data_set) - 1)

        image, label = data_set[idx]  # 获取图片和真实标签

        with torch.no_grad():  # 不计算梯度（节省内存）
            # 将图片输入模型获得预测,
            # unsqueeze 用于将二维数组图数据 [[1, 2], [3, 4]] 片转化为四维数组 [[[1, 2], [3, 4]]], 之前：[高度, 宽度, 通道数] -> [224, 224, 3] (这是一张图片)，之后：[批次大小, 高度, 宽度, 通道数] -> [1, 224, 224, 3] (这是一个包含一张图片的批次)
            # unsqueeze 参数代表在哪个梯度进行扩展dim=0: 在最前面加维度、dim=1: 在原来的第0维和第1维之间加维度、dim=-1: 在最后面加维度（非常常用）
            output = model(image.unsqueeze(0).to(device))
            pred = torch.argmax(output, dim=1).item()  # 找到概率最大的类别的索引的值

        axes[i].imshow(image.squeeze(), cmap="gray")  # 显示 UI 灰度图片
        axes[i].set_title(f"Predicted: {pred}")  # 设置 UI 标题显示预测结果
        axes[i].axis("off")  # 隐藏坐标轴

    # 显示 UI 图表
    plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")


def load_model(path):
    model = MnisModel()
    # map_location 确保模型权重加载到当前可用设备，避免在无 GPU 机器上加载 GPU 模型时报错
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
