import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime, sigmoid, sigmoid_prime
from losses import mse, mse_prime

# training data
#60,80,5,82
#70,75,7,94
#50,55,10,45
#40,56,7,42
#x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
#y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

x_train = np.array([[[60,80,5]], [[70,75,7]], [[50,55,10]], [[40,56,7]]])
y_train = np.array([[[82]], [[94]], [[45]], [[42]]])

# network
net = Network()
net.add(FCLayer(3, 4))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(4, 1))
net.add(ActivationLayer(sigmoid, sigmoid_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=10000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
