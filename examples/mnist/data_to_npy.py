import tensorflow as tf
import numpy as np
import msgpack
from tensorflow import keras

mnist = tf.keras.datasets.mnist
(images_train, labels_train), (images_test, labels_test) = mnist.load_data()

x = images_test[0]
y = labels_test[0]
print(y)
x = x.flatten() / 255.
x = x.astype(np.float32)

print(x.dtype, x.shape)
np.save('5.npy', x)
