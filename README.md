# Deep-Learning-with-Pytorch

Classifying Satellite Images with Convolutional Neural Networks

Satellite imagery provides a bird's eye view of changes happening on Earth's surface. With satellites collecting terabytes of new images daily, being able to automatically classify these images is key for practical applications. In this project, I have developed a convolutional neural network (CNN) model using PyTorch to categorize satellite images into different land use classes.

The dataset contained over 5000 64x64 pixel satellite images labeled into 4 classes - cloudy, desert, green area and water. After splitting the data 80/10/10 into train, validation and test sets, I built a CNN with the following architecture:

3 Conv2D layers with 32, 64 and 128 filters respectively

ReLU activations and 2x2 max pooling after each conv layer

2 more Conv2D layers with 256 filters

Flattened output fed into 3 fully connected layers

Final layer with 4 nodes for the classification outputs

The model was trained for 10 epochs using the Adam optimizer and cross entropy loss. The best validation accuracy achieved was 90.7%. When evaluated on the test set, the model obtained an accuracy of 85.4% and a loss of 0.46.
