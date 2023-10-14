import numpy as np

from network import Network
from conv_layer import ConvLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# training data
#x_train = [np.random.rand(10,10,1)]
#y_train = [np.random.rand(4,4,2)]
x_train = np.array([[[60,80,5]], [[70,75,7]], [[50,55,10]], [[40,56,7]], [[40,56,7]], [[40,75,7]]])
y_train = np.array([[[82]], [[94]], [[45]], [[42]], [[32]], [[42]]])
x_train = [array([[[0.44031167],[0.86070567],[0.08639991]],[[0.25391935],[0.24442995],[0.65916873]],[[0.71898313],[0.74089204],[0.64145705]]])]


# network
net = Network()
net.add(ConvLayer((10,10,1), (3,3), 1))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(ConvLayer((8,8,1), (3,3), 1))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(ConvLayer((6,6,1), (3,3), 2))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.3)

# test
out = net.predict(x_train)
print("predicted = ", out)
print("expected = ", y_train)
