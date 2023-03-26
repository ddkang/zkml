#
# A script for generating a backprop computational graph from forward
#

import argparse
import ast
from typing import Literal, Union
import msgpack
import numpy as np

class CircuitConfig():
    def __init__(self, starting_index):
        self.next_index = starting_index
        self.outp_to_grad = {}
        self.label_tensor_idx = None
        self.weights_update = None

    # Allocates an index for a gradient tensor and returns
    def new_gradient_tensor(self, tensor_idx):
        if tensor_idx in self.outp_to_grad:
            raise Exception("Tensor already allocated")
        self.outp_to_grad[tensor_idx] = self.next_index
        self.next_index += 1
        return self.outp_to_grad[tensor_idx]

    # Allocates an index for a tensor
    def new_tensor(self):
        new_index = self.next_index
        self.next_index += 1
        return new_index

    def new_label_tensor(self):
        if self.label_tensor_idx is not None:
            raise Exception("Label tensor already allocated")
        self.label_tensor_idx = self.next_index
        self.next_index += 1
        return self.label_tensor_idx

    # Allocates an index for a gradient tensor and returns
    def gradient_tensor_idx(self, tensor_idx):
        return self.outp_to_grad[tensor_idx]

# TODO: Put these in enums
NO_ACTIVATION = 0

SAME = 0
VALID = 1

CONV2D = 0
CONV2D_DEPTHWISE = 1

class Conv2D():
    def __init__(self, layer):
        params = layer['params']
        self.padding = params[1]
        self.activation_type = params[2]
        self.stride_h = params[3]
        self.stride_w = params[4]

    def backward(self, layer, transcript, config):
        inputs_idx, inputs_shape = layer['inp_idxes'][0], layer['inp_shapes'][0]
        weights_idx, weights_shape = layer['inp_idxes'][1], layer['inp_shapes'][1]
        bias_idx, bias_shape = layer['inp_idxes'][2], layer['inp_shapes'][2]
        output_idx, output_shape = layer['out_idxes'][0], layer['out_shapes'][0]

        permuted_inputs_idx = config.new_tensor()
        permutation = [3, 1, 2, 0]
        permuted_inputs_shape = [inputs_shape[p] for p in permutation]
        inputs_permute_layer = {
            'layer_type': 'Permute',
            'params': permutation,
            'inp_idxes': [inputs_idx],
            'out_idxes': [permuted_inputs_idx],
            'inp_shapes': [inputs_shape],
            'out_shapes': [permuted_inputs_shape],
            'mask': [],
        }
        transcript.append(inputs_permute_layer)

        permuted_outputs_idx = config.new_tensor()
        permuted_outputs_shape = [output_shape[p] for p in permutation]
        inputs_permute_layer = {
            'layer_type': 'Permute',
            'params': permutation,
            'inp_idxes': [config.gradient_tensor_idx(output_idx)],
            'out_idxes': [permuted_outputs_idx],
            'inp_shapes': [output_shape],
            'out_shapes': [permuted_outputs_shape],
            'mask': [],
        }
        transcript.append(inputs_permute_layer)


        dw_idx, dw_shape = config.new_tensor(), weights_shape
        dw_conv = {
            'layer_type': 'Conv2D',
            'params': [CONV2D, VALID, NO_ACTIVATION, self.stride_h, self.stride_w],
            'inp_idxes': [permuted_inputs_idx, permuted_outputs_idx],
            'out_idxes': [dw_idx],
            'inp_shapes': [permuted_inputs_shape, permuted_outputs_shape],
            'out_shapes': [dw_shape],
            'mask': [],
        }
        transcript.append(dw_conv)
        config.weights_update = dw_idx

        permutation = [3, 1, 2, 0]
        permutation_weights_idx = config.new_tensor()
        permutation_weights_shape = [weights_shape[p] for p in permutation]

        permute_weights = {
            'layer_type': 'Permute',
            'params': permutation,
            'inp_idxes': [weights_idx],
            'out_idxes': [permutation_weights_idx],
            'inp_shapes': [weights_shape],
            'out_shapes': [permutation_weights_shape],
            'mask': [],
        }
        transcript.append(permute_weights)

        rotated_weights_idx, rotated_weights_shape = config.new_tensor(), permutation_weights_shape
        rotate_layer = {
            'layer_type': 'Rotate',
            'params': [1, 2],
            'inp_idxes': [permutation_weights_idx],
            'out_idxes': [rotated_weights_idx],
            'inp_shapes': [permutation_weights_shape],
            'out_shapes': [rotated_weights_shape],
            'mask': [],
        }
        transcript.append(rotate_layer)

        padded_gradients_idx, padded_gradients_shape = config.new_tensor(), output_shape
        padded_gradients_shape[1] += (rotated_weights_shape[1] - 1) * 2 
        padded_gradients_shape[2] += (rotated_weights_shape[2] - 1) * 2
        pad_layer = {
            'layer_type': 'Pad',
            'params': [
                0, 0,
                rotated_weights_shape[1] - 1, rotated_weights_shape[1] - 1,
                rotated_weights_shape[2] - 1, rotated_weights_shape[2] - 1,
                0, 0
            ],
            'inp_idxes': [config.gradient_tensor_idx(output_idx)],
            'out_idxes': [padded_gradients_idx],
            'inp_shapes': [],
            'out_shapes': [],
            'mask': [],
        }
        transcript.append(pad_layer)

        dx_idx, dx_shape = config.new_gradient_tensor(inputs_idx), inputs_shape
        input_conv_layer = {
            'layer_type': 'Conv2D',
            'params': [CONV2D, VALID, NO_ACTIVATION, self.stride_h, self.stride_w],
            'inp_idxes': [padded_gradients_idx, rotated_weights_idx],
            'out_idxes': [dx_idx],
            'inp_shapes': [padded_gradients_shape, rotated_weights_shape],
            'out_shapes': [dx_shape],
            'mask': [],
        }
        transcript.append(input_conv_layer)

        permutation = [3, 1, 2, 0]
        permuted_dw_idx = config.new_tensor()
        permuted_dw_shape = [dw_shape[p] for p in permutation]

        permute_dw = {
            'layer_type': 'Permute',
            'params': permutation,
            'inp_idxes': [dw_idx],
            'out_idxes': [permuted_dw_idx],
            'inp_shapes': [dw_shape],
            'out_shapes': [permuted_dw_shape],
            'mask': [],
        }
        transcript.append(permute_dw)

        updated_weights_idx, updated_weights_shape = config.new_tensor(), dw_shape
        # Call a layer to update the outputs of the convolution
        update_weights_layer = {
            'layer_type': 'Update',
            'params': [],
            'inp_idxes': [weights_idx, permuted_dw_idx],
            'out_idxes': [updated_weights_idx],
            'inp_shapes': [weights_shape, permuted_dw_shape],
            'out_shapes': [updated_weights_shape],
            'mask': [],
        }
        # transcript.append(update_weights_layer)

class Softmax():
    def __init__(self, layer):
        return

    # TODO: Make this generalizable to all neural networks
    # (do not assume that softmax is the last layer, fused with CE-loss)
    def backward(self, layer, transcript, config):
        sub_layer = {
            'layer_type': 'Sub',
            'params': [],
            # y_hat - y
            'inp_idxes': [layer['out_idxes'][0], config.label_tensor_idx],
            'out_idxes': [config.new_gradient_tensor(layer['inp_idxes'][0])],
            'inp_shapes': [layer['out_shapes'][0], layer['out_shapes'][0]],
            'out_shapes': [layer['out_shapes'][0]],
            'mask': [],
        }
        transcript.append(sub_layer)

class AveragePool2D():
    def __init__(self, layer):
        return

    def backward(self, layer, transcript, config):
        # TODO: This is very model specific, must rewrite to be accurate
        # We just broadcast dx across 3 axes
        # 1 x 3 x 3 x 1 -> 1 x 1 x 1 x 1280

        div_idx = config.new_tensor()
        reshape_layer = {
            'layer_type': 'Broadcast',
            'params': [],
            'inp_idxes': [config.gradient_tensor_idx(layer['out_idxes'][0])],
            'out_idxes': [div_idx],
            'inp_shapes': [layer['out_shapes'][0]],
            'out_shapes': [layer['inp_shapes'][0]],
            'mask': [],
        }
        transcript.append(reshape_layer)

        out_idx = config.new_gradient_tensor(layer['inp_idxes'][0])
        out_shape = layer['inp_shapes'][0]

        div = {
            'layer_type': 'Div',
            'params': [layer['inp_shapes'][0][1] * layer['inp_shapes'][0][2]],
            'inp_idxes': [div_idx],
            'out_idxes': [out_idx],
            'inp_shapes': [out_shape],
            'out_shapes': [out_shape],
            'mask': [],
        }
        transcript.append(div)


class Reshape():
    def __init__(self, layer):
        return

    def backward(self, layer, transcript, config):
        reshape_layer = {
            'layer_type': 'Reshape',
            'params': [],
            'inp_idxes': [config.gradient_tensor_idx(layer['out_idxes'][0])],
            'out_idxes': [config.new_gradient_tensor(layer['inp_idxes'][0])],
            'inp_shapes': [layer['out_shapes'][0]],
            'out_shapes': [layer['inp_shapes'][0]],
            'mask': [],
        }
        transcript.append(reshape_layer)


def produce_graph():
    # Read msgpack file
    with open("examples/v2_1.0_224_truncated/model.msgpack", "rb") as data_file:
        byte_data = data_file.read()
    model = msgpack.unpackb(byte_data)

    # TODO: I'm unsure whether the circuit output is always the last indexed tensor
    softmax_output_index = int(np.max(
            [[out for out in layer['out_idxes']] for layer in model['layers']] + 
            [[inp for inp in layer['inp_idxes']] for layer in model['layers']]
    )[0])
    circuit_config = CircuitConfig(softmax_output_index + 1)
    circuit_config.new_label_tensor()

    transcript = []
    for layer in reversed(model['layers']):
        fetched_layer = None
        match layer['layer_type']:
            case "Conv2D":
                fetched_layer = Conv2D(layer)
            case "AveragePool2D":
                fetched_layer = AveragePool2D(layer)
            case "Softmax":
                fetched_layer = Softmax(layer)
            case _:
                fetched_layer = Reshape(layer)
        print(layer['layer_type'])
        fetched_layer.backward(layer, transcript, circuit_config)
        print('----------------')

    model['layers'] += transcript
    model['inp_idxes'].append(circuit_config.label_tensor_idx)
    model['out_idxes'] = [31]

    packed = msgpack.packb(model, use_bin_type=True)
    with open("./examples/train_graph/train.msgpack", 'wb') as f:
        f.write(packed)
    print(model.keys())
    return model

model = produce_graph()

print(model.keys())
model['tensors'] = ""
print(model['inp_idxes'], model['out_idxes'])
