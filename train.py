import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from network import Network


class Network:
    def __init__(self):
        # Initializing the hidden layers randomly
        self.w1 = torch.randn(784, 500)
        self.w2 = torch.randn(500, 250)
        self.w3 = torch.randn(250, 100)
        self.w4 = torch.randn(100, 10)

        self.learning_rate = 0.1 # setting learning rate

    def feedforward(self, X):
        # passing data through first hidden layer
        self.z1 = torch.matmul(X, self.w1) # sum of products
        self.a1 = self.sigmoid(self.z1) # activation function

        # passing data through second hidden layer
        self.z2 = torch.matmul(self.a1, self.w2)
        self.a2 = self.sigmoid(self.z2)

        # passing data through third hidden layer
        self.z3 = torch.matmul(self.a2, self.w3)
        self.a3 = self.sigmoid(self.z3)

        # passing data through fourth and final hidden layer
        self.z4 = torch.matmul(self.a3, self.w4)
        self.prediction = self.sigmoid(self.z4)

        return self.prediction

    def backpropagate(self, X, y, prediction):
        # calculating the delta for network output and target of MSE
        error = (2 * (y - prediction) / X.size(0)) * self.sigmoid_prime(prediction)
        self.w4 += self.learning_rate * torch.matmul(torch.t(self.a3), error) # updating the weights of fourth hidden layer

        error = torch.matmul(error, torch.t(self.w4)) * self.sigmoid_prime(self.a3) # delta of the weights of fourth hidden layer 
        self.w3 += self.learning_rate * torch.matmul(torch.t(self.a2), error) # updating weights of third hidden layer

        error = torch.matmul(error, torch.t(self.w3)) * self.sigmoid_prime(self.a2) # delta of third hidden layer
        self.w2 += self.learning_rate * torch.matmul(torch.t(self.a1), error) # updating second layer

        error = torch.matmul(error, torch.t(self.w2)) * self.sigmoid_prime(self.a1) # delta of second layer
        self.w1 += self.learning_rate * torch.matmul(torch.t(X), error) # updating first layer

    def train(self, X, y):
        prediction = self.feedforward(X) # input data to network for generating predictions
        self.backpropagate(X, y, prediction) # calculating error and updating the network based on predictions

    def sigmoid(self, x):
        """
        The sigmoid activation function.
        """
        return 1 / (1 + torch.exp(-x))

    def sigmoid_prime(self, x):
        """
        The derivative of the sigmoid.
        """
        return x * (1 - x)

    def save(self):
        torch.save(
            {
                "self.w1": self.w1,
                "self.w2": self.w2,
                "self.w3": self.w3,
                "self.w4": self.w4,
            },
            "network_weights.pkl",
        )
        print('Model weights saved as "network_weights.pkl"')

    def load(self):
        weights = torch.load("network_weights.pkl")
        self.w1 = weights["self.w1"]
        self.w2 = weights["self.w2"]
        self.w3 = weights["self.w3"]
        self.w4 = weights["self.w4"]
        print('Model weights loaded from "network_weights.pkl"')


def perform_validation():
    """
    Performs validation on the test and returns loss and accuracy.
    """
    val_data, val_target = next(iter(test_loader)) # retrieve data and target from loader.
    batch_size = val_target.size(0) # get the batch size.
    val_data = val_data.view(val_data.size(0), -1) # convert the images into vectors

    # converting the labels into a one-hot vector.
    val_target = val_target.unsqueeze(1) # Add another axis to the label vector and make it into a matrix
    val_target_onehot = torch.zeros(batch_size, 10) # Make a zero matrix.
    val_target_onehot.scatter_(1, val_target, 1) # Use scatter to add a one in the appropriate index in the one-hot vector

    val_prediction = net.feedforward(val_data) # forward pass the data through the network

    val_loss = torch.mean((val_target_onehot - val_prediction) ** 2 / batch_size) # calculate MSE loss
    validation_losses.append(val_loss) # append onto list for future plotting

    # calculate accuracy
    val_accuracy = torch.sum(
        torch.eq(val_target.squeeze(), torch.argmax(val_prediction, 1))
    ).item()
    validation_accuracies.append(val_accuracy) # append onto list for plotting

    return val_loss, (val_accuracy / val_data.size(0)) # return loss and accuracy


batch_size = 64
epochs = 20


# using the built-in mnist dataset
train_set = torchvision.datasets.MNIST(
    "dataset",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    ),
)
test_set = torchvision.datasets.MNIST(
    "dataset",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    ),
)

# creating data loader for train and test set
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

net = Network() # initializing network

# empty lists for storing diagnostic information
train_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []

# Training loop
for epoch in range(epochs):
    print(f"Starting epoch {epoch+1} of {epochs}")
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = target.size(0) # getting current batch size
        data = data.view(data.size(0), -1) # converting data instances which are matrices into vectors

        target = target.unsqueeze(1) # adding another axis to the target label vector for generating one-hot vectors
        target_onehot = torch.zeros(batch_size, 10) # creating a zero matrix
        target_onehot.scatter_(1, target, 1) # putting a 1 in the appropriate index of target_onehot using scatter function

        net.train(data, target_onehot) # doing a forward pass and a backward pass and updating weights

    # calculating losses and accuracy and validation tests
    prediction = net.feedforward(data)
    train_loss = torch.mean((target_onehot - prediction) ** 2 / batch_size)
    train_losses.append(train_loss)

    train_accuracy = torch.sum(
        torch.eq(target.squeeze(), torch.argmax(prediction, 1))
    ).item()
    training_accuracies.append(train_accuracy)

    val_loss, val_accuracy = perform_validation()

    print(f"Train Loss: {train_loss} Validation Loss: {val_loss}")
    print(
        f"Train Accuracy: {train_accuracy/data.size(0) * 100}% Validation Accuracy: {val_accuracy*100}%"
    )

    net.save() # saving the current weights of the network
    print()

# Testing loop (we did it in the file eval)
#accuracy = 0
#for batch_idx, (data, target) in enumerate(test_loader):
#    batch_size = target.size(0)
#    data = data.view(data.size(0), -1)
#
#    target = target.unsqueeze(1)
#    target_onehot = torch.zeros(batch_size, 10)
#    target_onehot.scatter_(1, target, 1)

#   prediction = net.feedforward(data)

#    accuracy += torch.sum(
#        torch.eq(target.squeeze(), torch.argmax(prediction, 1))
#    ).item()

#print(f"Testing accuracy: {accuracy}/{len(test_set)} {(accuracy/len(test_set)) * 100}%")

# plotting data
plt.figure(1)
plt.plot(train_losses, label="Train Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.title('Loss')
plt.xlabel("Epochs")
plt.legend()
plt.savefig("training-validation-loss.png", dpi=300)

plt.figure(2)
plt.plot(training_accuracies, label='Train Accuracy')
plt.plot(validation_accuracies, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('training-validation-accuracy.png', dpi=300)
