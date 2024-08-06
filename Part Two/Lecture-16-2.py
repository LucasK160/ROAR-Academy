## This is course material for Introduction to Modern Artificial Intelligence
## Example code: mlp.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

# Load dependencies
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Create data
linearSeparableFlag = False
x_bias = 6

def toy_2D_samples(x_bias ,linearSeparableFlag):
    label1 = np.array([[1, 1],[-1,-1]])
    label2 = np.array([[1, -1],[-1,1]])

    if linearSeparableFlag:

        samples1 = np.random.multivariate_normal([5+x_bias, 0], [[1, 0],[0, 1]], 100)
        samples2 = np.random.multivariate_normal([-5+x_bias, 0], [[1, 0],[0, 1]], 100)

        samples = np.concatenate((samples1, samples2 ), axis =0)
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'rx')
        plt.show()

    else:
        samples1 = np.random.multivariate_normal([5+x_bias, 5], [[1, 0],[0, 1]], 50)
        samples2 = np.random.multivariate_normal([-5+x_bias, -5], [[1, 0],[0, 1]], 50)
        samples3 = np.random.multivariate_normal([-5+x_bias, 5], [[1, 0],[0, 1]], 50)
        samples4 = np.random.multivariate_normal([5+x_bias, -5], [[1, 0],[0, 1]], 50)

        samples = np.concatenate((samples1, samples2, samples3, samples4 ), axis =0)
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'bo')
        plt.plot(samples3[:, 0], samples3[:, 1], 'rx')
        plt.plot(samples4[:, 0], samples4[:, 1], 'rx')
        plt.show()

    label1 = np.array([[1, 1],[-1,-1]])
    label2 = np.array([[1, -1],[-1,1]])
    labels1 = np.repeat(label1, 100, axis = 0)
    labels2 = np.repeat(label2, 100, axis = 0)
    labels = np.concatenate((labels1, labels2 ), axis =0)
    return samples, labels

samples, labels = toy_2D_samples(x_bias ,linearSeparableFlag)

# Split training and testing set

randomOrder = np.random.permutation(200)
trainingX = samples[randomOrder[0:100], :]
trainingY = labels[randomOrder[0:100], :]
testingX = samples[randomOrder[100:200], :]
testingY = labels[randomOrder[100:200], :]

# model = Sequential()
# model.add(Dense(4, input_shape=(2,), activation='sigmoid', use_bias=True))
# # model.add(Dense(4, input_shape=(2,), activation='relu', use_bias=True))
# model.add(Dense(2, activation='softmax' ))
# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])

model = Sequential()
model.add(Dense(2, input_shape=(2,), activation='softmax', use_bias=True))  # Two perceptrons for XOR
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])

model.fit(trainingX, trainingY, epochs=800, batch_size=3, verbose=1, validation_split=0.2)

# score = model.evaluate(testingX, testingY, verbose=0)
score = 0
for i in range(100):
    predict_x=model.predict(np.array([testingX[i,:]])) 
    estimate=np.argmax(predict_x,axis=1)

    if np.array_equal(testingY[i], [1, 1]) and estimate == 0:
        score += 1
        plt.plot(testingX[i, 0], testingX[i, 1], 'bo')
    elif np.array_equal(testingY[i], [-1, -1]) and estimate == 1:
        score += 1
        plt.plot(testingX[i, 0], testingX[i, 1], 'rx')
    elif np.array_equal(testingY[i], [1, -1]) and estimate == 2:
        score += 1
        plt.plot(testingX[i, 0], testingX[i, 1], 'go')
    elif np.array_equal(testingY[i], [-1, 1]) and estimate == 3:
        score += 1
        plt.plot(testingX[i, 0], testingX[i, 1], 'yx')

print('Test accuracy:', score/100)
plt.show()
