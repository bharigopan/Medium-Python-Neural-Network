import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime, sigmoid, sigmoid_prime
from losses import mse, mse_prime

# training data
x_train = np.array([[[60,80,5]], [[70,75,7]], [[50,55,10]], [[40,56,7]]])
y_train = np.array([[[82]], [[94]], [[45]], [[42]]])

# network
net = Network()
net.add(FCLayer(3, 3))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(sigmoid, sigmoid_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1, learning_rate=0.01)

# test
out = net.predict(x_train)
print(out)
