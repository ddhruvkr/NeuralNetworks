# this is an implementation of a simple neural netork from scratch only using numpy
# for now it takes as input points which for 3 clusters and it tries to guess where the new point belongs
# for now 3 clusters are centered at (0,-2), (2,2) and (2,-2)
# it gives the output as values close to 0 for (0,-2), 0.5 for (2,2) and around 1 for (2,-2)


import numpy as np
import matplotlib.pyplot as plt

samplesPerClass = 500

X1 = np.random.randn(samplesPerClass, 2) + np.array([0, -2])
X2 = np.random.randn(samplesPerClass, 2) + np.array([2,2])
X3 = np.random.randn(samplesPerClass, 2) + np.array([2,-2])
X = np.vstack([X1, X2, X3])
#print("shape of X")
#print(X)
#print(X.shape)
Y = np.array([0]*samplesPerClass + [0.5]*samplesPerClass + [1]*samplesPerClass)
Y = Y.T
Y = np.reshape(Y, (samplesPerClass*3,1))
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()
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
hiddenLayer = 4
outputLayer = 1

W1 = np.random.randn(inputLayer, hiddenLayer)
#print("shape of W1")
#print (W1.shape)
b1 = np.random.randn(hiddenLayer,1)
#print("shape of b")
#print (b1.shape)
W2 = np.random.randn(hiddenLayer, outputLayer)
#print("shape of W2")
#print (W2.shape)
b2 = np.random.randn(outputLayer,1)
#print("shape of b2")
#print (b2.shape)

def activate(x, deriv=False):
	if deriv==True:
		return (x*(1-x))
	return 1/(1+np.exp(-x))

def forwardPropagation(X, W1, b1, W2, b2):

	layer1 = activate(X.dot(W1) + b1.T)
	layer2 = activate(layer1.dot(W2) + b2.T)
	#layer1 = activate(X.dot(W1))
	#layer2 = activate(layer1.dot(W2))
	return layer1, layer2

def backwardPropagation(Y, layer2, layer1, layer0, W2, b2, W1, b1, i):
	#print("shape of layer2")
	#print (layer2.shape)
	#print(layer2)
	#print("shape of layer1")
	#print (layer1.shape)
	#print(layer1)
	#print(Y)
	#print(layer2)

	l2Error = Y - layer2
	if i % 1000 == 0:
		print ("Error:" + str(np.mean(np.abs(l2Error))))
	#print("shape of l2Error")
	#print (l2Error.shape)
	#print(l2Error)
	layer2Gradient = l2Error * activate(layer2, deriv=True)
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
	W2 += 0.001 * layer1.T.dot(layer2Gradient)
	#print(b2.shape)
	b2 += 0.001 * layer2Gradient.mean()
	#print(b2.shape)
	W1 += 0.001 * layer0.T.dot(layer1Gradient)
	b1 += 0.001 * layer1Gradient.mean()
	return W1, b1, W2, b2

for i in range(10000):
	layer1, layer2 = forwardPropagation(X, W1, b1, W2, b2)
	W1, b1, W2, b2 = backwardPropagation(Y, layer2, layer1, X, W2, b2, W1, b1, i)

test = np.array([0,-2])
_, ans = forwardPropagation(test, W1, b1, W2, b2)
print(ans)
test = np.array([2,2])
_, ans = forwardPropagation(test, W1, b1, W2, b2)
print(ans)
test = np.array([2,-2])
_, ans = forwardPropagation(test, W1, b1, W2, b2)
print(ans)