# A converter for training data
# Performs the conversion npy -> msgpack
# TODO: Ensure that training works with models that take in multiple input shapes
# 
# Shortcut: 
# `python3 python/training_converter.py --input_shapes 7,7,320 --input_idxes 1,0 --output training_data/inputs.msgpack --labels_output training_data/labels.msgpack`
#

import argparse
import ast
import numpy as np
import msgpack
import os

NUM_LOADS = 10

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_shapes', type=str, required=True)
  parser.add_argument('--input_idxes', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)
  parser.add_argument('--labels_output', type=str, required=True)

  TRAINING_DIRECTORY = './data/pre_last_conv/flowers/train'
  args = parser.parse_args()

  input_shapes = ast.literal_eval(args.input_shapes)
  input_idxes = ast.literal_eval(args.input_idxes)

  loaded = 0
  tensors = []
  labels = []
  num_classes = os.listdir(TRAINING_DIRECTORY)

  for file_name in os.listdir(TRAINING_DIRECTORY):
    label = int(file_name[:-4])
    data_array = np.load(TRAINING_DIRECTORY + '/' + file_name)

    input_shape = input_shapes[0]
    input_index = input_idxes[0]

    for idx in range(data_array.shape[0]):
      tensors.append({
        'idx': input_index,
        'shape': input_shape,
        'data': data_array[idx].flatten().tolist(),
      })
      # represent the label as a one hot encoding
      one_hot = np.zeros(len(num_classes))
      one_hot[label] = 1
      labels.append({
        'label': list(one_hot)
      })
      loaded += 1

    if loaded == NUM_LOADS:
      break

  packed_inputs = msgpack.packb(tensors, use_bin_type=True)
  packed_labels = msgpack.packb(labels, use_bin_type=True)

  with open(args.output, 'wb') as f:
    f.write(packed_inputs)

  with open(args.labels_output, 'wb') as f:
    f.write(packed_labels)

if __name__ == '__main__':
  main()
