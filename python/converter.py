import argparse
import tensorflow as tf
import numpy as np
import tflite
import msgpack

def get_shape(interpreter: tf.lite.Interpreter, tensor_idx):
  if tensor_idx == -1:
    return []
  tensor = interpreter.get_tensor(tensor_idx)
  return list(tensor.shape)


def get_inputs(op: tflite.Operator):
  idxes = op.InputsAsNumpy()
  idxes = idxes.tolist()
  idxes = list(filter(lambda x: x != -1, idxes))
  return idxes

class Converter:
  def __init__(self, model_path, scale_factor, k, num_cols):
    self.model_path = model_path
    self.scale_factor = scale_factor
    self.k = k
    self.num_cols = num_cols

    self.interpreter = tf.lite.Interpreter(
      model_path=self.model_path,
      experimental_preserve_all_tensors=True
    )
    self.interpreter.allocate_tensors()

    with open(self.model_path, 'rb') as f:
      buf = f.read()
      self.model = tflite.Model.GetRootAsModel(buf, 0)
    self.graph = self.model.Subgraphs(0)


  def _convert_add(self, op, generated_tensors: set):
    # Get inputs
    inputs = get_inputs(op)
    print(generated_tensors)
    print('Add inputs: ', inputs)
    if len(inputs) != 2:
      raise RuntimeError('Add must have 2 inputs')

    # If both tensors are generated, do nothing
    print(inputs[0] in generated_tensors, inputs[1] in generated_tensors)
    if (inputs[0] in generated_tensors) and (inputs[1] in generated_tensors):
      return ('Add', [])

    nb_generated = (inputs[0] in generated_tensors) + (inputs[1] in generated_tensors)
    if nb_generated != 1:
      raise RuntimeError('Add must have 1 generated tensor')

    # Check if there are any negative infinities
    const_tensor = self.interpreter.get_tensor(inputs[0]) if inputs[0] not in generated_tensors else self.interpreter.get_tensor(inputs[1])
    if np.any(const_tensor == -np.inf):
      # Ensure that the constant tensor is all -inf and 0
      if not np.all(np.logical_or(np.isneginf(const_tensor), const_tensor == 0)):
        raise RuntimeError('Add constant tensor must be -inf and 0 only')
      mask = (const_tensor == -np.inf).astype(np.int64)
      params = [len(mask.shape)] + list(mask.shape)
      params += mask.flatten().tolist()
      return ('MaskNegInf', params)
    else:
      return ('Add', [])

  def to_dict(self, inps, start_layer, end_layer):
    interpreter = self.interpreter
    model = self.model
    graph = self.graph

    input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    for i, inp in enumerate(inps):
      interpreter.set_tensor(input_details[i]['index'], inp)
    interpreter.invoke()

    # Get layers
    generated_tensor_idxes = set()
    for inp in input_details:
      generated_tensor_idxes.add(inp['index'])

    layers = []
    keep_tensors = set()
    for op_idx in range(graph.OperatorsLength()):
      op = graph.Operators(op_idx)
      op_code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()

      # Skip generated tensors
      for output in op.OutputsAsNumpy():
        generated_tensor_idxes.add(output)

      if op_idx < start_layer:
        continue
      if op_idx > end_layer:
        break

      # Keep the input tensors
      for input in op.InputsAsNumpy():
        keep_tensors.add(input)

      # AvgPool2D
      if op_code == tflite.BuiltinOperator.AVERAGE_POOL_2D:
        layer_type = 'AveragePool2D'
        op_opt = op.BuiltinOptions()
        opt = tflite.Pool2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        params = [opt.FilterHeight(), opt.FilterWidth(), opt.StrideH(), opt.StrideW()]
      # Conv2D
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
      # DepthwiseConv2D
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
      # Fully connected
      elif op_code == tflite.BuiltinOperator.FULLY_CONNECTED:
        layer_type = 'FullyConnected'
        op_opt = op.BuiltinOptions()
        opt = tflite.FullyConnectedOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.FusedActivationFunction() != tflite.ActivationFunctionType.NONE and opt.FusedActivationFunction() != tflite.ActivationFunctionType.RELU6:
          raise NotImplementedError('Only ReLU6 and None supported')
        activation = 1 if opt.FusedActivationFunction() == 3 else 0
        params = [activation]
      elif op_code == tflite.BuiltinOperator.BATCH_MATMUL:
        layer_type = 'BatchMatMul'
        params = []

      ## Arithmetic
      # Add
      elif op_code == tflite.BuiltinOperator.ADD:
        layer_type, params = self._convert_add(op, generated_tensor_idxes)
      # Mul
      elif op_code == tflite.BuiltinOperator.MUL:
        layer_type = 'Mul'
        params = []
      # Sub
      elif op_code == tflite.BuiltinOperator.SUB:
        layer_type = 'Sub'
        params = []
      # Pad
      elif op_code == tflite.BuiltinOperator.PAD:
        layer_type = 'Pad'
        tensor_idx = op.Inputs(1)
        tensor = interpreter.get_tensor(tensor_idx).flatten().astype(np.int64)
        params = tensor.tolist()
      # Softmax
      elif op_code == tflite.BuiltinOperator.SOFTMAX:
        layer_type = 'Softmax'
        params = []
      # Mean
      elif op_code == tflite.BuiltinOperator.MEAN:
        layer_type = 'Mean'
        tensor_idx = op.Inputs(1)
        tensor = interpreter.get_tensor(tensor_idx).flatten().astype(np.int64)
        if len(tensor) != 1:
          raise NotImplementedError('Only mean over one axis is supported')
        params = tensor.tolist()
      # Squared difference
      elif op_code == tflite.BuiltinOperator.SQUARED_DIFFERENCE:
        layer_type = 'SquaredDifference'
        params = []

      # Pointwise
      elif op_code == tflite.BuiltinOperator.RSQRT:
        layer_type = 'Rsqrt'
        params = []
      elif op_code == tflite.BuiltinOperator.LOGISTIC:
        layer_type = 'Logistic'
        params = []

      # The following are no-ops in the sense that they don't change the tensor
      # However, we need to pass along the right tensors
      # The param says which input to pass along
      elif op_code == tflite.BuiltinOperator.SHAPE:
        layer_type = 'Noop'
        params = [0]
      elif op_code == tflite.BuiltinOperator.GATHER:
        layer_type = 'Noop'
        params = [0]
      elif op_code == tflite.BuiltinOperator.REDUCE_PROD:
        # TODO: not sure if this is in general a no-op
        layer_type = 'Noop'
        params = [0]
      elif op_code == tflite.BuiltinOperator.PACK:
        layer_type = 'Noop'
        params = [0]
      elif op_code == tflite.BuiltinOperator.CONCATENATION:
        # FIXME: This is not in general a no-op
        layer_type = 'Noop'
        params = [0]

      ## Shape
      elif op_code == tflite.BuiltinOperator.RESHAPE:
        layer_type = 'Reshape'
        params = []
      elif op_code == tflite.BuiltinOperator.TRANSPOSE:
        layer_type = 'Transpose'
        params = get_shape(interpreter, op.Inputs(0)) + interpreter.get_tensor(op.Inputs(1)).flatten().astype(np.int64).tolist()
      else:
        op_name = None
        for attr in dir(tflite.BuiltinOperator):
          if not attr.startswith('__'):
            if getattr(tflite.BuiltinOperator, attr) == op_code:
              op_name = attr
        raise NotImplementedError('Unsupported operator: {}, {}'.format(op_code, op_name))

      inp_idxes = get_inputs(op)
      # FIXME: hack for testing
      if op_idx == 99:
        mask = [0, 1]
      else:
        mask = []
      layers.append({
        'layer_type': layer_type,
        'inp_idxes': inp_idxes,
        'inp_shapes': [get_shape(interpreter, inp_idx) for inp_idx in inp_idxes],
        'out_idxes': [op.Outputs(i) for i in range(op.OutputsLength())],
        'out_shapes': [get_shape(interpreter, op.Outputs(i)) for i in range(op.OutputsLength())],
        'params': params,
        'mask': mask,
      })
    print(layers)
    print()


    # Get tensors
    print('keep tensors:', keep_tensors)
    tensors = []
    for tensor_idx in range(graph.TensorsLength()):
      if tensor_idx not in keep_tensors:
        continue
      if tensor_idx in generated_tensor_idxes:
        print(f'skipping generated tensor: {format(tensor_idx)}, {graph.Tensors(tensor_idx).Name()}')
        continue

      tensor = graph.Tensors(tensor_idx)
      shape = []
      for i in range(tensor.ShapeLength()):
        shape.append(int(tensor.Shape(i)))
      if shape == []:
        shape = [1]

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
    print()
    print(d['layers'][-1])
    print(d['out_idxes'])
    # d['out_idxes'] = [14]
    print(d.keys())
    print(d['out_idxes'])
    return d

  def to_msgpack(self, inps, start_layer, end_layer):
    d = self.to_dict(inps, start_layer, end_layer)
    return msgpack.packb(d, use_bin_type=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True)
  parser.add_argument('--input_shape', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)
  parser.add_argument('--scale_factor', type=int, default=2**16)
  parser.add_argument('--k', type=int, default=19)
  parser.add_argument('--num_cols', type=int, default=6)
  parser.add_argument('--start_layer', type=int, default=0)
  parser.add_argument('--end_layer', type=int, default=10000)
  args = parser.parse_args()

  converter = Converter(args.model, args.scale_factor, args.k, args.num_cols)
  inp_shape = [int(x) for x in args.input_shape.split(',')]
  inps = [np.zeros(inp_shape, dtype=np.float32)]
  packed = converter.to_msgpack(inps, start_layer=args.start_layer, end_layer=args.end_layer)

  with open(args.output, 'wb') as f:
    f.write(packed)

if __name__ == '__main__':
  main()
