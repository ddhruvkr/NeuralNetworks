# this is an implementation of a simple neural network with 2 hidden layers from scratch (only using numpy)
# for now it takes as input points which for 3 clusters and it tries to guess where the new point belongs
# for now 3 clusters are centered at (0,-20), (20,20) and (20,-20)
# it is a multi classifier so it gives the output in the form of 3 nodes, highlighting the node which is the appropriate cluster


import numpy as np
import matplotlib.pyplot as plt

samplesPerClass = 200

X1 = np.random.randn(samplesPerClass, 2) + np.array([0, -20])
X2 = np.random.randn(samplesPerClass, 2) + np.array([20,20])
X3 = np.random.randn(samplesPerClass, 2) + np.array([20,-20])
X = np.vstack([X1, X2, X3])

Y = np.array([1,0,0])
for i in range(samplesPerClass-1):
	Y = np.vstack((Y, [1,0,0]))
for i in range(samplesPerClass):
	Y = np.vstack((Y, [0,1,0]))
for i in range(samplesPerClass):
	Y = np.vstack((Y, [0,0,1]))

inputLayer=2
hiddenLayer1 = 4
hiddenLayer2 = 4
outputLayer = 3

W1 = np.random.randn(inputLayer, hiddenLayer1)
b1 = np.random.randn(hiddenLayer1,1)
W2 = np.random.randn(hiddenLayer1, hiddenLayer2)
b2 = np.random.randn(hiddenLayer2,1)
W3 = np.random.randn(hiddenLayer2, outputLayer)
b3 = np.random.randn(outputLayer,1)

def activate(x, deriv=False):
	if deriv==True:
		return (x*(1-x))
	return 1/(1+np.exp(-x))

def forwardPropagation(X, W1, b1, W2, b2, W3, b3):
	layer1 = activate(X.dot(W1) + b1.T)
	layer2 = activate(layer1.dot(W2) + b2.T)
	layer3 = activate(layer2.dot(W3) + b3.T)
	return layer1, layer2, layer3

def backwardPropagation(Y, layer3, layer2, layer1, layer0, W3, b3, W2, b2, W1, b1, i):
	layer3Error = Y - layer3
	if i % 1000 == 0:
		print ("Error:" + str(np.mean(np.abs(layer3Error))))
	layer3Gradient = layer3Error * activate(layer3, deriv=True)
	layer2Error = layer3Gradient.dot(W3.T)
	layer2Gradient = layer2Error * activate(layer2, deriv=True)
	layer1Error = layer2Gradient.dot(W2.T)
	layer1Gradient = layer1Error * activate(layer1, deriv=True)
	W3 += 0.1 * layer2.T.dot(layer3Gradient)
	b3 += 1 * np.matrix(layer3Gradient.mean(axis=0)).T
	W2 += 0.1 * layer1.T.dot(layer2Gradient)
	b2 += 1 * np.matrix(layer2Gradient.mean(axis=0)).T
	W1 += 0.1 * layer0.T.dot(layer1Gradient)
	b1 += 1 * np.matrix(layer1Gradient.mean(axis=0)).T
	return W1, b1, W2, b2, W3, b3

for i in range(10000):
	layer1, layer2, layer3 = forwardPropagation(X, W1, b1, W2, b2, W3, b3)
	W1, b1, W2, b2, W3, b3 = backwardPropagation(Y, layer3, layer2, layer1, X, W3, b3, W2, b2, W1, b1, i)

test = np.array([1,-20])
_, _, ans = forwardPropagation(test, W1, b1, W2, b2, W3, b3)
print(ans)
test = np.array([15,15])
_, _, ans = forwardPropagation(test, W1, b1, W2, b2, W3, b3)
print(ans)
test = np.array([20,-20])
_, _,ans = forwardPropagation(test, W1, b1, W2, b2, W3, b3)
print(ans)