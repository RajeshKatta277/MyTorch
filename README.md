# MyTorch
MyTorch is my custom-implemented deep learning framework inspired by PyTorch. Its a simplified version of pytroch implemented everything from scratch using numpy. It is designed to provide a hands-on learning experience and a simplified environment for understanding the inner workings of deep neural networks. MyTorch offers a range of functionalities commonly found in deep learning frameworks, including various layers(linear layers,activationlayers, cnns etc) , activation functions, loss functions, and optimization algorithms. This project aims to empower users with a clear understanding of how neural networks function and how to build them from scratch using Python and NumPy. 

# key features and functionalities:

 ## 1. Layers:
 MyTorch provides different types of layers such as fullyconnectedlayers, Convolutional etc. Each layer type is implemented to perform forward and backward passes, enabling users to create complex neural network architectures. The Reshape Layer in MyTorch is designed to transform the output from Conv(CNN) layers into a flattened shape, suitable for passing the data to Fully Connected (FCLayer) layers. 

**Usage:**  Mytorch.Layers contains different types of layers like FClayer, conv layer etc
example:
```
Mytorch.Layers.FCLayer(input_size,output_size) or 
Mytorch.Layers.Conv(kernel_size,input_channels,output_channels) etc
```



## 2. Activation Functions: 
The framework includes popular activation functions like ReLU, Sigmoid, Tanh, Leaky ReLU, etc., allowing users to introduce non-linearity into their neural networks. little note that we treat activation layer as seperate layer to keep things modular. 

**Usage :** Mytorch.ActivatoinFunctions contains different types of activation functions include ReLU, LeakyReLU, Sigmoid and Tanh. 
```
Mytorch.ActivationFunctions.Sigmoid() or 
Mytorch.ActivationFunctions.ReLU() etc 
```

## 3. Network: 
We can crate a neural network with different layers using this network class from mytorch. we can create network like net=Network() and then we can add layers to the network. like fc1,activation1,fc2,activation2 or conv1,activation1,conv2,activaition2,reshapelayer,fc1, activation1 etc.


## 4. Loss Functions:
Mytorch provides Various loss functions such as Cross Entropy, Mean Squared Error, and Binary Cross Entropy are implemented to evaluate the performance of models during training and to compute derivatives of loss with respect to each weight in each layer.   

**Usage:** Mytorch.LossFunctions contains different loss functions classes and each loss function class consists of two methods, .Loss() method computes error in predictions, .backward() method computes gradients of loss with respect to each weights in each layer.
Example:
```
Mytorch.LossFunctions.CrossEntropy().Loss(true_values,predictions) computes and returns overall loss or error in the predictions.
Mytorch.LossFunctions.CrossEntropy().backward(network,true_values,predictions) computes derivatives of loss function with respect to each weight in each layer.
```


## 5. Optimization Algorithms: 
MyTorch offers different optimization algorithms like Gradient Descent, Gradient Descent with Momentum, RMSprop, and Adam for adjusting model parameters and minimizing the loss function. 

**Usage:** Mytorch.Optimizers contains different optimization algorithms like gradient descent,grdient descent with momentum, rmsprop and adam. each optimizer consists of two methods .step() updates weights and .zero_grad() clears the accumulations like momentum etc you can simply zero_grad after performing step() so that moments from previous epoch doesn't effect current epoch learning. 
```
Mytorch.Optimizers.GradientDescent(network.layers,learning_rate) or 
Mytorch.Optimizers.Adam(network.layers,beta1,beta2,learning_rate) etc.

```
## Example:

**Importing:**
```
import Mytorch
import Mytorch.Network.Network as Network 
from Mytorch.ActivationFunctions import Sigmoid,Tanh,ReLU,LeakyReLU,Softmax 
from Mytorch.Layers import FCLayer,Conv,Reshape 
from Mytorch.LossFunctions import BinaryCrossEntropy, CrossEntropy, MeanSquaredError
from Mytorch.Optimizers import GradientDescent, GradientDescentWithMomentum, RMSProp, Adam 

```
**Network:**
```
net=Network()
net.add_layers(Conv(kernel_size=3,input_depth=1,output_depth=5))
net.add_layers(Sigmoid())
net.add_layers(Conv(kernel_size=3,input_depth=5,output_depth=10))
net.add_layers(Sigmoid())
net.add_layers(Reshape((10,24,24),(10*24*24,1)))
net.add_layers(FCLayer(10*24*24,100))
net.add_layers(Sigmoid())
net.add_layers(FCLayer(100,10))
net.add_layers(Sigmoid())
```

**Training:**
```
learning_rate= 0.001
optimizer=GradientDescent(net.layers,learning_rate=learning_rate)
error=BinaryCrossEntropy()
epochs=5
for epoch in range(epochs):
    epoch_loss=[]
    acc=[]
    for images,labels in train_loader:
        data=images.reshape(images.shape[0],1,28,28)
        targets=to_one_hot(labels)
        data=data.numpy()
        targets=targets.numpy()
        outputs=net.predict(data)
        loss=error.Loss(targets,outputs)#returns loss
        epoch_loss.append(loss)
        acc.append(accuracy(np.argmax(outputs,axis=1),np.argmax(targets,axis=1)))
        error.backward(net,targets,outputs)#computes derivatives of loss with respect to each weight in each layer
        optimizer.step()# updates weights
        
    print(f"epoch:{epoch+1}, loss:{np.mean(epoch_loss)} , accuracy:{np.mean(acc)*100}%")
```
**output:**
```
epoch:1, loss:0.1297362714460865 , accuracy:80.28833333333333%
epoch:2, loss:0.056575114539481795 , accuracy:91.85%
epoch:3, loss:0.04424243689721972 , accuracy:93.47%
epoch:4, loss:0.03710513431695464 , accuracy:94.61333333333334%
epoch:5, loss:0.03215961753225156 , accuracy:95.37166666666667%
```
