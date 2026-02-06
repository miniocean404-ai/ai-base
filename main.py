from mnist import train


def train_mnist():
    train_loader, test_loader, train_dataset, test_dataset = train.download_mnist()
    model = train.MnisModel()
    train.train(model, train_loader)
    train.predict(model, test_dataset)
    train.save_model(model, "model.pth")
    # model = run.load_model("model.pth")
    # run.predict(model, test_dataset)


def main():
    train_mnist()


if __name__ == "__main__":
    main()
