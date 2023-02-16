import argparse
import tensorflow as tf
import numpy as np
import tflite
import msgpack

class Converter:
  def __init__(self, model_path, scale_factor, k, num_cols):
    self.model_path = model_path
    self.scale_factor = scale_factor
    self.k = k
    self.num_cols = num_cols

  def to_dict(self, inps):
    interpreter = tf.lite.Interpreter(
      model_path=self.model_path,
      experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, inp in enumerate(inps):
      interpreter.set_tensor(input_details[i]['index'], inp)

    with open(self.model_path, 'rb') as f:
      buf = f.read()
      model = tflite.Model.GetRootAsModel(buf, 0)
    graph = model.Subgraphs(0)

    # Get layers
    generated_tensor_idxes = set()
    for inp in input_details:
      generated_tensor_idxes.add(inp['index'])

    layers = []
    for op_idx in range(graph.OperatorsLength()):
      op = graph.Operators(op_idx)
      op_code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()

      # Skip generated tensors
      for output in op.OutputsAsNumpy():
        generated_tensor_idxes.add(output)

      if op_code == tflite.BuiltinOperator.AVERAGE_POOL_2D:
        layer_type = 'AveragePool2D'
        op_opt = op.BuiltinOptions()
        opt = tflite.Pool2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        params = [opt.FilterHeight(), opt.FilterWidth(), opt.StrideH(), opt.StrideW()]
      elif op_code == tflite.BuiltinOperator.CONV_2D:
        layer_type = 'Conv2D'
        op_opt = op.BuiltinOptions()
        opt = tflite.Conv2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.DilationHFactor() != 1 or opt.DilationWFactor() != 1:
          raise NotImplementedError('Dilation is not supported')
        if opt.FusedActivationFunction() != tflite.ActivationFunctionType.NONE and opt.FusedActivationFunction() != tflite.ActivationFunctionType.RELU6:
          raise NotImplementedError('Only ReLU6 and None supported')
        # 0 is Conv2D
        activation = 1 if opt.FusedActivationFunction() == 3 else 0
        params = \
          [0] + \
          [opt.Padding()] + \
          [activation] + \
          [opt.StrideH(), opt.StrideW()]
      elif op_code == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
        layer_type = 'Conv2D'
        op_opt = op.BuiltinOptions()
        opt = tflite.DepthwiseConv2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.DilationHFactor() != 1 or opt.DilationWFactor() != 1:
          raise NotImplementedError('Dilation is not supported')
        if opt.FusedActivationFunction() != tflite.ActivationFunctionType.NONE and opt.FusedActivationFunction() != tflite.ActivationFunctionType.RELU6:
          raise NotImplementedError('Only ReLU6 and None supported')
        # 1 is DepthwiseConv2D
        activation = 1 if opt.FusedActivationFunction() == 3 else 0
        params = \
          [1] + \
          [opt.Padding()] + \
          [activation] + \
          [opt.StrideH(), opt.StrideW()]
      elif op_code == tflite.BuiltinOperator.ADD:
        layer_type = 'Add'
        params = []
      elif op_code == tflite.BuiltinOperator.PAD:
        layer_type = 'Pad'
        # FIXME: the padding input is a tensor, not a parameter. Fix in rust
        tensor_idx = op.Inputs(1)
        tensor = interpreter.get_tensor(tensor_idx).flatten().flatten().astype(np.int64)
        params = tensor.tolist()
      elif op_code == tflite.BuiltinOperator.SOFTMAX:
        continue
        print('softmax')
      elif op_code == tflite.BuiltinOperator.RESHAPE:
        continue
        print('reshape')
        print(op.InputsLength())
      else:
        op_name = None
        for attr in dir(tflite.BuiltinOperator):
          if not attr.startswith('__'):
            if getattr(tflite.BuiltinOperator, attr) == op_code:
              op_name = attr
        raise NotImplementedError('Unsupported operator: {}, {}'.format(op_code, op_name))

      layers.append({
        'layer_type': layer_type,
        'inp_idxes': [op.Inputs(i) for i in range(op.InputsLength())],
        'out_idxes': [op.Outputs(i) for i in range(op.OutputsLength())],
        'params': params,
      })
    print(layers)


    # Get tensors
    tensors = []
    for tensor_idx in range(graph.TensorsLength()):
      if tensor_idx in generated_tensor_idxes:
        print(f'skipping generated tensor: {format(tensor_idx)}, {graph.Tensors(tensor_idx).Name()}')
        continue
      tensor = graph.Tensors(tensor_idx)
      shape = []
      # If the tensor is generated, skip it
      if tensor.Buffer() == 0:
        continue
      for i in range(tensor.ShapeLength()):
        shape.append(int(tensor.Shape(i)))

      tensor_data = interpreter.get_tensor(tensor_idx).flatten()
      if tensor.Type() == tflite.TensorType.FLOAT32:
        tensor_data = (tensor_data * self.scale_factor).round().astype(np.int64)
      elif tensor.Type() == tflite.TensorType.INT32:
        tensor_data = tensor_data.astype(np.int64)
      else:
        raise NotImplementedError('Unsupported tensor type: {}'.format(tensor.Type()))

      tensors.append({
        'idx': tensor_idx,
        'shape': shape,
        'data': tensor_data.tolist(),
      })
      # print(tensor_idx, tensor.Type(), tensor.Name(), tensors[-1]['shape'])
      # print(np.abs(tensor_data).max())

    d = {
      'global_sf': self.scale_factor,
      'k': self.k,
      'num_cols': self.num_cols,
      'inp_idxes': [inp['index'] for inp in input_details],
      # 'out_idxes': [out['index'] for out in output_details],
      'out_idxes': layers[-1]['out_idxes'],
      'layers': layers,
      'tensors': tensors,
    }
    print(d['layers'][-1])
    print(d['out_idxes'])
    # d['out_idxes'] = [14]
    print(d.keys())
    print(d['out_idxes'])
    return d

  def to_msgpack(self, inps):
    d = self.to_dict(inps)
    return msgpack.packb(d, use_bin_type=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True)
  parser.add_argument('--input_shape', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)
  parser.add_argument('--scale_factor', type=int, default=2**16)
  parser.add_argument('--k', type=int, default=19)
  parser.add_argument('--num_cols', type=int, default=6)
  args = parser.parse_args()

  converter = Converter(args.model, args.scale_factor, args.k, args.num_cols)
  inp_shape = [int(x) for x in args.input_shape.split(',')]
  inps = [np.zeros(inp_shape, dtype=np.float32)]
  packed = converter.to_msgpack(inps)

  with open(args.output, 'wb') as f:
    f.write(packed)

if __name__ == '__main__':
  main()