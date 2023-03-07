import tensorflow as tf
import os 
import numpy as np

class LastTwoLayers(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name = name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.conv1 = tf.keras.layers.Conv2D(
            1280,
            (1, 1),
            activation='relu6',
            padding='same',
            input_shape=(1, 3, 3, 112)
        )
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(3, 3),
            padding='valid',
            strides=(1, 1)
        )
        self.conv2 = tf.keras.layers.Conv2D(
            1001,
            (1, 1),
            padding='valid',
            input_shape=(1, 1, 1, 1280)
        )
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.conv1(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = tf.reshape(x, [1, 1001])
        x = self.softmax(x)
        return x

my_sequential_model = LastTwoLayers(name="the_model")
my_sequential_model.compile(optimizer='sgd', loss='categorical_crossentropy')
x = np.random.random((1, 3, 3, 112))
my_sequential_model.predict(x)

converter = tf.lite.TFLiteConverter.from_keras_model(my_sequential_model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
