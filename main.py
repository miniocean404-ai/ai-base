from mnist import train
from utils import DEVICE
from utils.file import create_dir


def train_mnist():
    print(f"使用设备: {DEVICE}")
    train_loader, test_loader, train_dataset, test_dataset = train.download_mnist()
    model = train.MnisModel()
    # 将模型迁移到 GPU/CPU
    model.to(DEVICE)

    train.train(model, train_loader)
    train.predict(model, test_dataset)

    train_model_path = "train/mnist_model.pth"
    create_dir(train_model_path)

    train.save_model(model, train_model_path)
    # model = train.load_model(train_model_path)
    # train.predict(model, test_dataset)


def main():
    train_mnist()


if __name__ == "__main__":
    main()
