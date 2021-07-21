import torch
import torchvision
from torchvision import transforms
from network import Network
from hw1_931161715_931158984_train import Network


def evaluate_hw1():
    """
    The evaluation function. It will load the trained weights
    and forward pass the test set and then calculate accuracy.
    """
    batch_size = 64

    test_set = torchvision.datasets.MNIST(
        "dataset",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        ),
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True
    )

    net = Network()
    net.load()

    accuracy = 0

    for data, target in test_loader:
        batch_size = target.size(0)
        data = data.view(data.size(0), -1)

        target = target.unsqueeze(1)
        target_onehot = torch.zeros(batch_size, 10)
        target_onehot.scatter_(1, target, 1)

        prediction = net.feedforward(data)

        accuracy += torch.sum(
            torch.eq(target.squeeze(), torch.argmax(prediction, 1))
        ).item()

    print(f"Testing accuracy: {accuracy}/{len(test_set)} {(accuracy/len(test_set)) * 100}%")


if __name__ == "__main__":
    evaluate_hw1()
