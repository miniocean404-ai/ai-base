import torch
import torchvision
from torchvision import transforms


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
