#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as pl
# load data
train_data = np.load("train_data.npy")
train_label = np.load("train_label.npy")
test_data = np.load("test_data.npy")
test_label = np.load("test_label.npy")

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)


# In[2]:


# 使用MinMaxScaler对数据进行归一化处理
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)


# In[3]:


# create a activation class
# for each time, we can initiale a activation function object with one specific function
# for example: f = Activation("tanh")  means we create a tanh activation function.
# you can define more activation functions by yourself, such as relu!

class Activation(object):
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)   
        return 1.0 - a**2
    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x) 
        return  a * (1 - a )
    
    def __init__(self,activation='tanh'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv


# In[4]:


class HiddenLayer(object):    
    def __init__(self,n_in, n_out,
                 activation_last_layer='tanh',activation='tanh', W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input=None
        self.activation=Activation(activation).f
        
        # activation deriv of last layer
        self.activation_deriv=None
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).f_deriv

        # we randomly assign small values for the weights as the initiallization
        self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
        )
        # if activation == 'logistic':
        #     self.W *= 4

        # we set the size of bias as the size of output dimension
        self.b = np.zeros(n_out,)
        
        # we set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)
        
    
    # the forward and backward progress (in the hidden layer level) for each training epoch
    # please learn the week2 lec contents carefully to understand these codes. 
    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input=input
        return self.output
    
    def backward(self, delta, output_layer=False):         
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta


# In[5]:


class MLP:
    """
    """ 

    # for initiallization, the code will create all layers automatically based on the provided parameters.     
    def __init__(self, layers, activation=[None,'tanh','tanh']):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """        
        ### initialize layers
        self.layers=[]
        self.params=[]
        
        self.activation=activation
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1]))

    # forward progress: pass the information through the layers and out the results of final output layer
    def forward(self,input):
        for layer in self.layers:
            output=layer.forward(input)
            input=output
        return output

    # define the objection/loss function, we use mean sqaure error (MSE) as the loss
    # you can try other loss, such as cross entropy.
    # when you try to change the loss, you should also consider the backward formula for the new loss as well!
    def criterion_MSE(self,y,y_hat):
        activation_deriv=Activation(self.activation[-1]).f_deriv
        # MSE
        error = y-y_hat
        loss=error**2
        # calculate the MSE's delta of the output layer
        delta=-error*activation_deriv(y_hat)    
        # return loss and delta
        return loss,delta

    # backward progress  
    def backward(self,delta):
        delta=self.layers[-1].backward(delta,output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta)

    # update the network weights after backward.
    # make sure you run the backward function before the update function!    
    def update(self,lr):
        for layer in self.layers:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b

    # define the training function
    # it will return all losses within the whole training process.
    def fit(self,X,y,learning_rate=0.1, epochs=100,batch_size=16):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """ 
        X=np.array(X)
        y=np.array(y)
        to_return = np.zeros(epochs)
        
        for k in range(epochs):
            loss=np.zeros(X.shape[0])
            for it in range(X.shape[0]):
                i=np.random.randint(X.shape[0])
                
                # forward pass
                y_hat = self.forward(X[i])
                
                # backward pass
                loss[it],delta=self.criterion_MSE(y[i],y_hat)
                self.backward(delta)
                y
                # update
                self.update(learning_rate)
            to_return[k] = np.mean(loss)
        return to_return

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i,:])
        return output


# In[6]:


### Try different MLP models
nn = MLP([2,3,1], [None,'logistic','tanh'])
input_data = train_data[:,0:2]
output_data = train_data[:,2]


# In[7]:


def fit_batch(X, Y, model, learning_rate):
    """Fit one batch of training data"""
    # Forward pass
    A = model.forward(X)
    loss = model.loss(Y, A)
    
    # Backward pass
    dA = model.loss_derivative(Y, A)
    model.backward(dA)
    
    # Update parameters
    model.update_params(learning_rate)
    
    return loss

def fit(X, Y, model, learning_rate, epochs, batch_size):
    """Fit the model to the training data"""
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            
            loss = fit_batch(X[start:end], Y[start:end], model, learning_rate)
            epoch_loss += loss
        
        losses.append(epoch_loss / num_samples)
        
    return losses


# In[9]:


# create an instance of the MLP class


# define new input data
new_input_data = np.array([[0.2, 0.1], [0.8, 0.9], [0.5, 0.3]])

# use the trained network to predict the output for some new input data
predicted_output = nn.predict(new_input_data)

# print the predicted output
print(predicted_output)

nn = MLP([2, 5, 1], [None, 'tanh', 'tanh'])

# train the network on the input data and output data
MSE = nn.fit(input_data, output_data, learning_rate=0.001, epochs=50)

# print the final loss
print('loss: %f' % MSE[-1])

# use the trained network to predict the output for some new input data
predicted_output = net.predict(new_input_data)


# In[28]:


pl.figure(figsize=(15,4))
pl.plot(MSE)
pl.grid()


# In[14]:


def evaluate_mlp(layers, lr):
    model = MLP(layers, activation=[None,'tanh','tanh'])
    for i in range(1000):
        # forward pass
        y_pred = model.forward(train_data)
        # calculate loss
        loss = np.mean((train_label - y_pred) ** 2)
        # backward pass
        delta = y_pred - train_label
        for layer in reversed(model.layers):
            delta = layer.backward(delta)
        # update parameters
        for layer in model.layers:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b
    # predict test data
    y_test_pred = model.forward(test_data)
    # calculate test loss
    test_loss = np.mean((test_label - y_test_pred) ** 2)
    return test_loss


# In[20]:


# create an instance of the MLP class


# define new input data
new_input_data = np.array([[0.2, 0.1], [0.8, 0.9], [0.5, 0.3]])

# use the trained network to predict the output for some new input data
predicted_output = nn.predict(new_input_data)

# print the predicted output
print(predicted_output)

nn = MLP([2, 5, 1], [None, 'tanh', 'tanh'])

# train the network on the input data and output data
MSE = nn.fit(input_data, output_data, learning_rate=0.001, epochs=50)

# print the final loss
print('loss: %f' % MSE[-1])

# use the trained network to predict the output for some new input data
predicted_output = nn.predict(new_input_data)


# In[ ]:





# In[ ]:





# In[25]:





# In[ ]:





# In[ ]:




