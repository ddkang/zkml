import tensorflow as tf
import os 
import numpy as np

interpreter = tf.lite.Interpreter(
    model_path=f'./testing/circuits/v2_1.0_224.tflite'
)
interpreter.allocate_tensors()

NAME_TO_TENSOR = {}
for tensor_details in interpreter.get_tensor_details():
    NAME_TO_TENSOR[tensor_details['name']] = tensor_details
Wc = interpreter.get_tensor(NAME_TO_TENSOR['Const_71']['index'])
Bc = interpreter.get_tensor(NAME_TO_TENSOR['MobilenetV2/Conv_1/Conv2D_bias']['index'])

class LastTwoLayers(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name = name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.conv1 = tf.keras.layers.Conv2D(
            1280,
            (1, 1),
            activation='relu6',
            padding='same',
            input_shape=(1, 7, 7, 320)
        )
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(7, 7),
            padding='valid',
            strides=(1, 1)
        )
        self.conv2 = tf.keras.layers.Conv2D(
            102,
            (1, 1),
            padding='valid',
            input_shape=(1, 1, 1, 1280)
        )
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.conv1(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = tf.reshape(x, [1, 102])
        x = self.softmax(x)
        return x

my_sequential_model = LastTwoLayers(name="the_model")
my_sequential_model.compile(optimizer='sgd', loss='categorical_crossentropy')

x = np.random.random((1, 7, 7, 320))

my_sequential_model.predict(x)

my_sequential_model.conv1.set_weights([np.transpose(Wc, [1,2,3,0]), Bc])
# x, y, chan_in, chan_out

# 1 batch, 7 height, 320 width, 7 channels
# 7 height, 1 width, 7 channels, 1280 out channels
# 1 height, 320 width, 1280 out channels

# 1 Batch, 7 height, 7 width, 320 channels () [This is the input to the layer]
# 1 Batch, 7 height, 7 width, 1280 channels (dout) [This is the output of another layer]

# We want to transform this so that we rotate the input by 90 degrees

W = np.zeros([1, 1, 1280, 102])

my_sequential_model.conv2.set_weights([
    W,
    np.zeros([102])
])

converter = tf.lite.TFLiteConverter.from_keras_model(my_sequential_model)
tflite_model = converter.convert()

with open('./examples/v2_1.0_224_truncated/v2_1.0_224_truncated.tflite', 'wb') as f:
  f.write(tflite_model)
