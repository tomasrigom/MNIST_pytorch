# MNIST in pytorch
Classification of MNIST images using the pytorch library. Experiment ran on a kaggle notebook.

### Network Characteristics

The main features of the network are:

- **Two convolutional-max pooling layers** with kernel sizes 5 and 2 respectively, each outputting 32 and 64 different channels, and Rectified Linear Unit (ReLU) activation functions. The second layer also includes a **dropout layer** to avoid overfitting, with a 10% probability of dropout given to each unit (neuron).
- **A linear ReLU layer with dropout**, having the same 10% probability of dropout for each neuron. This layer has 50 hidden units.
- **An output Softmax-Log Likelihood layer**, implemented in pytorch as a linear layer which is then fed to nn.CrossEntropyLoss as loss function, which performs the calculation for us.

### Training

The aforementioned network object is then fed to a **Stochastic Gradient Descent (SGD) algorithm** as optimizer object, with initial **learning rate 0.01 sheduled to decay by factors of 0.1 when the accuracy on the validation data plateaus for 5 epochs**, a **weight decay parameter lambda of 0.01**, a **momentum co-efficient of 0.3** and a patience parameter (for early stopping) of 30 epochs.

The training is performed on a GPU, made available byonline computing company kaggle.

### Results

The model finished training due to early stopping at epoch 69, yielding a training data accuracy of 97.71%, validation data accuracy of 97.84%, and a **final test data accuracy of 98.38%**. A plot is also provided with the evolution of these values.