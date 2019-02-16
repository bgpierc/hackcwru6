from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #fuck it
os.environ["CUDA_VISIBLE_DEVICES"]="0" #gpu

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure() 
plt.imshow(train_images[0]) #show the first raw image
plt.colorbar()
plt.grid(False)
#plt.show()

train_images = train_images / 255.0 #scale to a range of 0-1, (intensity)

test_images = test_images / 255.0

plt.figure(figsize=(10,10)) #show the first 25 greyscale images
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixel
    keras.layers.Dense(128, activation=tf.nn.relu), #training layers
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
