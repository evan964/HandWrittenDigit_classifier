# HandWrittenDigit_classifier

Feed Forward Neural Network implementation for MNIST dataset using only torch tensor manipulations without built-in pytorch function.


## Model Architecture, Hyperparameters, Optimization:

The model consists of four fully-connected linear layers. There sizes and order is as
follows:

1. FC Layer [input: 784, output: 500]
2. FC Layer [input: 500, output: 250]
3. FC Layer [input: 250, output: 100]
4. FC Layer [input: 100, output: 10]


Activation Function: Sigmoid
Loss Function: Mean Squared Error
Optimization Function: Stochastic Gradient Descent
Learning Rate: 0.1
Training Epochs: 20
Batch Size: 64
Approximate Accuracy: 85%

We know that the accuracy would be better if we had used Softmax for activation
function and Cross Entropy for the loss function, but we get an accuracy of more than
75 without this, so we preferred let like this.

## Training Procedure:
1. The data instances received from the data loader is converted to a vector.
2. The target labels received from the data loader are converted to a one-hot
vector.
3. Both the data and target labels are fed into the model for training. The forward
pass and the backward pass are implemented in the “network.py” script.
4. After an epoch is passed the loss and accuracy are calculated by performing a
forward pass and then calculating the MSE loss.
5. The accuracy and loss are also calculated on a batch of the test set.
6. When the model is done training, the weights are saved in a pickle file.
7. Then the model is tested on the test set and the accuracy is calculated.
8. Finally, the plots are generated and saved.

## Results
![image](https://user-images.githubusercontent.com/87755953/126473755-7eb53008-f24f-4816-bfc9-76bffdec633f.png)
![image](https://user-images.githubusercontent.com/87755953/126473783-ea540d90-4846-420d-a89a-807edf38265e.png)

Approximate Accuracy of the Test Set: 87.8%

##Conclusion:

During the course of training the network, I tried several hyperparameter values. Initially,
I set the learning rate to 1 which resulted in the network diverging considerably. Then I
set it to 0.001 which made the learning slower and the model did not converge.
Eventually, after trying multiple values I set it to 0.1 which resulted in the model
converging and having good accuracy. I’ve found that the learning rate has a substantial
impact on the performance of the neural network and should be tuned carefully.
Another important hyperparameter is the number of epochs. I’ve found that increasing
the number of epochs is almost always beneficial for the network. Initially, I wanted to
save time and so I trained for just five epochs which were not enough iterations for the
model to converge. So I kept on increasing it until at 20 epochs I was getting
satisfactory results. I am sure if I increase the epochs even more I can further increase
the accuracy.
Batch size has minimal impact on the network performance instead it affects the
memory consumption of the training. If the batch size is small then the network will
consume less memory.


