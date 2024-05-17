from tensorflow import keras
import tensorflow as tf
import os
import onnx
import tf2onnx

print("Loading model 🕒....")

model_file_name = "mobilenet_v2_1.0_224"
model = keras.applications.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

print("Model Loading complete ✅ ")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()

# Create the directory if it doesn't exist
output_dir = f"generated/mobilenet_v2_1.0_224"
if not os.path.exists(output_dir):
    print(f"Folder not found, creating folder {output_dir}")
    os.makedirs(output_dir)

open (f"generated/{model_file_name}/model.tflite" , "wb") .write(tfmodel)

print(f"Model saved as generated/{model_file_name}/model.tflite ✅ ")

print("Converting model to ONNX format 🕒....")

# Convert the Keras model to ONNX
spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the ONNX model
onnx_output_path = f"generated/{model_file_name}/model.onnx"
onnx.save_model(onnx_model, onnx_output_path)

print(f"Model saved as {onnx_output_path} ✅ ")


