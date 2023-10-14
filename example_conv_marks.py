import numpy as np

from network import Network
from conv_layer import ConvLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# training data
#x_train = [np.random.rand(10,10,1)]
#y_train = [np.random.rand(4,4,2)]
#x_train = np.array([[[60,80,5]], [[70,75,7]], [[50,55,10]], [[40,56,7]], [[40,56,7]], [[40,75,7]]])
#y_train = np.array([[[82]], [[94]], [[45]], [[42]], [[32]], [[42]]])
x_train = np.array([[[60],[80],[5]],[[70],[75],[7]],[[50],[55],[10]]])
y_train = np.array([[[82]], [[94]], [[45]]])

# network
net = Network()
net.add(ConvLayer((3,3,1), (2,2), 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.add(ConvLayer((2,2,1), (2,2), 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.3)

# test
out = net.predict(x_train)
print("predicted = ", out)
print("expected = ", y_train)
