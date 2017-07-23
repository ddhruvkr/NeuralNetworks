# this is an implementation of a simple neural network with 2 hidden layers from scratch (only using numpy)
# for now it takes as input points which for 3 clusters and it tries to guess where the new point belongs
# for now 3 clusters are centered at (0,-20), (20,20) and (20,-20)
# it gives the output as values close to 0 for (0,-20), 0.5 for (20,20) and around 1 for (20,-20)
# the case where it gives multiple classification is still to done, i.e output layer > 1


import numpy as np
import matplotlib.pyplot as plt

samplesPerClass = 200

X1 = np.random.randn(samplesPerClass, 2) + np.array([0, -20])
X2 = np.random.randn(samplesPerClass, 2) + np.array([20,20])
X3 = np.random.randn(samplesPerClass, 2) + np.array([20,-20])
X = np.vstack([X1, X2, X3])
#print("shape of X")
#print(X)
#print(X.shape)
Y = np.array([0]*samplesPerClass + [0.5]*samplesPerClass + [1]*samplesPerClass)
Y = np.reshape(Y, (samplesPerClass*3,1))
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
#plt.show()
#print("shape of Y")
#print (Y.shape)
'''X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
Y = np.array([[0],
            [1],
            [1],
            [0]])
inputLayer = 3'''
inputLayer=2
hiddenLayer1 = 4
hiddenLayer2 = 4
outputLayer = 1

W1 = np.random.randn(inputLayer, hiddenLayer1)
#print("shape of W1")
#print (W1.shape)
b1 = np.random.randn(hiddenLayer1,1)
#print("shape of b")
#print (b1.shape)
W2 = np.random.randn(hiddenLayer1, hiddenLayer2)
#print("shape of W2")
#print (W2.shape)
b2 = np.random.randn(hiddenLayer2,1)
#print("shape of b2")
#print (b2.shape)
W3 = np.random.randn(hiddenLayer2, outputLayer)
#print("shape of W2")
#print (W2.shape)
b3 = np.random.randn(outputLayer,1)
#print("shape of b2")
#print (b2.shape)

def activate(x, deriv=False):
	if deriv==True:
		return (x*(1-x))
	return 1/(1+np.exp(-x))

def forwardPropagation(X, W1, b1, W2, b2, W3, b3):

	layer1 = activate(X.dot(W1) + b1.T)
	layer2 = activate(layer1.dot(W2) + b2.T)
	layer3 = activate(layer2.dot(W3) + b3.T)
	#layer1 = activate(X.dot(W1))
	#layer2 = activate(layer1.dot(W2))
	return layer1, layer2, layer3

def backwardPropagation(Y, layer3, layer2, layer1, layer0, W3, b3, W2, b2, W1, b1, i):
	#print("shape of layer2")
	#print (layer2.shape)
	#print(layer2)
	#print("shape of layer1")
	#print (layer1.shape)
	#print(layer1)
	#print(Y)
	#print(layer3)

	layer3Error = Y - layer3
	#print(layer3Error)
	if i % 1000 == 0:
		print ("Error:" + str(np.mean(np.abs(layer3Error))))
	layer3Gradient = layer3Error * activate(layer3, deriv=True)
	#print("shape of l2Error")
	#print (l2Error.shape)
	#print(l2Error)
	layer2Error = layer3Gradient.dot(W3.T)
	layer2Gradient = layer2Error * activate(layer2, deriv=True)
	#print("shape of layer2Gradient")
	#print (layer2Gradient.shape)
	#print (layer2Gradient)
	layer1Error = layer2Gradient.dot(W2.T)
	#print("shape of layer1Error")
	#print (layer1Error.shape)
	#print (layer1Error)
	layer1Gradient = layer1Error * activate(layer1, deriv=True)
	#print("shape of layer1Gradient")
	#print (layer1Gradient.shape)
	#print (layer1Gradient)
	W3 += 0.1 * layer2.T.dot(layer3Gradient)
	b3 += 1 * np.matrix(layer3Gradient.mean(axis=0)).T
	W2 += 0.1 * layer1.T.dot(layer2Gradient)
	#print(b2.shape)
	b2 += 1 * np.matrix(layer2Gradient.mean(axis=0)).T
	#b2 += 0.1 * layer2Gradient.mean()
	#print(b2.shape)
	W1 += 0.1 * layer0.T.dot(layer1Gradient)
	b1 += 1 * np.matrix(layer1Gradient.mean(axis=0)).T
	#b1 += 0.1 * layer1Gradient.mean()
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