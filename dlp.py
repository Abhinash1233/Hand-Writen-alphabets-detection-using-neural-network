#trainning a model that classify the images of numbers using mnit dataset of images
#using deep neural network
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
(X_test, y_test),(x_train,y_train)= keras.datasets.mnist.load_data()
x_train = x_train/255
X_test = X_test/255
x_train_flattened=x_train.reshape(len(x_train),28*28)
X_test_flattened = X_test.reshape(len(X_test),28*28)
x_train_flattened.shape
X_test_flattened.shape

#creating the neural network
model = keras.Sequential([
    #adding hidden layer
    keras.layers.Dense(100, input_shape=(784,),activation = 'relu'),
    keras.layers.Dense(10,activation = 'sigmoid') #last layer
])

model.compile(
    optimizer  ='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

model.fit(x_train_flattened,y_train,epochs = 15)

model.evaluate(X_test_flattened)
y_predicted =model.predict(X_test_flattened)
y_prediction = tf.argmax(y_predicted, axis=1)
print("predicted value = ",np.argmax(y_predicted[1000]))
print("Actual value = ",y_test[1000])

# to get the over view that how our model is working well we can us confusion matrix
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_prediction)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot = True, fmt= 'd')
plt.xlabel('predicted')
plt.ylabel('Truth')

plt.show()