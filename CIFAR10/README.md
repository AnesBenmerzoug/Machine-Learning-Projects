# CIFAR-10 Network
A Neural Network used to classify the images of the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) Dataset.
The Network is composed of two BLSTM layers and a two Fully Connected layer.
The first fully connected layer combines the 3 channels of the input image into a single channel.
Then, the BLSTM layers process the image in the horizontal and vertical direction respectively.
Finally, The second fully connected layer takes the output of both BLSTM layers and gives a class conditional
probability for the 10 classes in the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) Dataset using a softmax non-linearity.
