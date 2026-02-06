from mnist import train
from mnist.train import device


def train_mnist():
    print(f"使用设备: {device}")
    train_loader, test_loader, train_dataset, test_dataset = train.download_mnist()
    model = train.MnisModel().to(device)  # 将模型迁移到 GPU/CPU
    train.train(model, train_loader)
    train.predict(model, test_dataset)
    train.save_model(model, "model.pth")
    # model = train.load_model("model.pth")
    # train.predict(model, test_dataset)


def main():
    train_mnist()


if __name__ == "__main__":
    main()
