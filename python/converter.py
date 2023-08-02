import argparse
import ast
from typing import Literal, Union
import tensorflow as tf
import numpy as np
import tflite
import msgpack

def get_shape(interpreter: tf.lite.Interpreter, tensor_idx):
  if tensor_idx == -1:
    return []
  tensor = interpreter.get_tensor(tensor_idx)
  return list(tensor.shape)

def handle_numpy_or_literal(inp: Union[np.ndarray, Literal[0]]):
  if isinstance(inp, int):
    return np.array([inp])
  return inp

def get_inputs(op: tflite.Operator):
  idxes = handle_numpy_or_literal(op.InputsAsNumpy())
  idxes = idxes.tolist()
  idxes = list(filter(lambda x: x != -1, idxes))
  return idxes

class Converter:
  def __init__(
      self, model_path, scale_factor, k, num_cols, num_randoms, use_selectors, commit,
      expose_output
    ):
    self.model_path = model_path
    self.scale_factor = scale_factor
    self.k = k
    self.num_cols = num_cols
    self.num_randoms = num_randoms
    self.use_selectors = use_selectors
    self.commit = commit
    self.expose_output = expose_output

    self.interpreter = tf.lite.Interpreter(
      model_path=self.model_path,
      experimental_preserve_all_tensors=True
    )
    self.interpreter.allocate_tensors()

    with open(self.model_path, 'rb') as f:
      buf = f.read()
      self.model = tflite.Model.GetRootAsModel(buf, 0)
    self.graph = self.model.Subgraphs(0)


  def valid_activations(self):
    return [
      tflite.ActivationFunctionType.NONE,
      tflite.ActivationFunctionType.RELU,
      tflite.ActivationFunctionType.RELU6,
    ]

  def _convert_add(self, op: tflite.Operator, generated_tensors: set):
    # Get params
    op_opt = op.BuiltinOptions()
    if op_opt is None:
      raise RuntimeError('Add options is None')
    opt = tflite.AddOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)
    params = [opt.FusedActivationFunction()]

    # Get inputs
    inputs = get_inputs(op)
    print(generated_tensors)
    print('Add inputs: ', inputs)
    if len(inputs) != 2:
      raise RuntimeError('Add must have 2 inputs')

    # If both tensors are generated, do nothing
    print(inputs[0] in generated_tensors, inputs[1] in generated_tensors)
    if (inputs[0] in generated_tensors) and (inputs[1] in generated_tensors):
      return ('Add', params)

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
      return ('Add', params)


  def to_dict(self, start_layer, end_layer):
    interpreter = self.interpreter
    model = self.model
    graph = self.graph
    if graph is None:
      raise RuntimeError('Graph is None')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for inp_detail in input_details:
      inp = np.zeros(inp_detail['shape'], dtype=inp_detail['dtype'])
      interpreter.set_tensor(inp_detail['index'], inp)
    # for i, inp in enumerate(inps):
    #   interpreter.set_tensor(input_details[i]['index'], inp)
    interpreter.invoke()

    # Get layers
    generated_tensor_idxes = set()
    for inp in input_details:
      generated_tensor_idxes.add(inp['index'])

    layers = []
    keep_tensors = set()
    adjusted_tensors = {}
    for op_idx in range(graph.OperatorsLength()):
      op = graph.Operators(op_idx)
      if op is None:
        raise RuntimeError('Operator is None')
      model_opcode = model.OperatorCodes(op.OpcodeIndex())
      if model_opcode is None:
        raise RuntimeError('Operator code is None')
      op_code = model_opcode.BuiltinCode()

      # Skip generated tensors
      for output in handle_numpy_or_literal(op.OutputsAsNumpy()):
        generated_tensor_idxes.add(output)

      if op_idx < start_layer:
        continue
      if op_idx > end_layer:
        break

      # Keep the input tensors
      for input in handle_numpy_or_literal(op.InputsAsNumpy()):
        keep_tensors.add(input)

      # AvgPool2D
      if op_code == tflite.BuiltinOperator.AVERAGE_POOL_2D:
        layer_type = 'AveragePool2D'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('AvgPool2D options is None')
        opt = tflite.Pool2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        params = [opt.FilterHeight(), opt.FilterWidth(), opt.StrideH(), opt.StrideW()]
      elif op_code == tflite.BuiltinOperator.MAX_POOL_2D:
        layer_type = 'MaxPool2D'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('MaxPool2D options is None')
        opt = tflite.Pool2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.Padding() == tflite.Padding.SAME:
          raise NotImplementedError('SAME padding is not supported')
        if opt.FusedActivationFunction() != tflite.ActivationFunctionType.NONE:
          raise NotImplementedError('Fused activation is not supported')
        params = [opt.FilterHeight(), opt.FilterWidth(), opt.StrideH(), opt.StrideW()]
      # FIXME: hack for Keras... not sure why this isn't being converted properly
      elif op_code == tflite.BuiltinOperator.CUSTOM:
        layer_type = 'Conv2D'
        activation = 0
        weights = self.interpreter.get_tensor(op.Inputs(1))
        weights = np.transpose(weights, (3, 0, 1, 2))
        weights = (weights * self.scale_factor).round().astype(np.int64)
        adjusted_tensors[op.Inputs(1)] = weights
        params = [0, 1, activation, 1, 1]
      # Conv2D
      elif op_code == tflite.BuiltinOperator.CONV_2D:
        layer_type = 'Conv2D'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('Conv2D options is None')
        opt = tflite.Conv2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.DilationHFactor() != 1 or opt.DilationWFactor() != 1:
          raise NotImplementedError('Dilation is not supported')
        if opt.FusedActivationFunction() not in self.valid_activations():
          raise NotImplementedError('Unsupported activation function at layer {op_idx}')
        # 0 is Conv2D
        params = \
          [0] + \
          [opt.Padding()] + \
          [opt.FusedActivationFunction()] + \
          [opt.StrideH(), opt.StrideW()]
      # DepthwiseConv2D
      elif op_code == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
        layer_type = 'Conv2D'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('DepthwiseConv2D options is None')
        opt = tflite.DepthwiseConv2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.DilationHFactor() != 1 or opt.DilationWFactor() != 1:
          raise NotImplementedError('Dilation is not supported')
        if opt.FusedActivationFunction() not in self.valid_activations():
          raise NotImplementedError('Unsupported activation function at layer {op_idx}')
        # 1 is DepthwiseConv2D
        params = \
          [1] + \
          [opt.Padding()] + \
          [opt.FusedActivationFunction()] + \
          [opt.StrideH(), opt.StrideW()]
      # Fully connected
      elif op_code == tflite.BuiltinOperator.FULLY_CONNECTED:
        layer_type = 'FullyConnected'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('Fully connected options is None')
        opt = tflite.FullyConnectedOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.FusedActivationFunction() not in self.valid_activations():
          raise NotImplementedError(f'Unsupported activation function at layer {op_idx}')
        params = [opt.FusedActivationFunction()]
      elif op_code == tflite.BuiltinOperator.BATCH_MATMUL:
        layer_type = 'BatchMatMul'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('BatchMatMul options is None')
        opt = tflite.BatchMatMulOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.AdjX() is True: raise NotImplementedError('AdjX is not supported')
        params = [int(opt.AdjX()), int(opt.AdjY())]

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
        sub_val = interpreter.get_tensor(op.Inputs(1))
        # TODO: this is a bit of a hack
        if np.any(np.isin(sub_val, 10000)):
          layer_type = 'MaskNegInf'
          mask = (sub_val == 10000).astype(np.int64)
          params = [len(mask.shape)] + list(mask.shape)
          params += mask.flatten().tolist()
        else:
          layer_type = 'Sub'
          params = []
      # Div
      elif op_code == tflite.BuiltinOperator.DIV:
        # Implement division as multiplication by the inverse
        layer_type = 'Mul'
        div_val = interpreter.get_tensor(op.Inputs(1))
        if type(div_val) != np.float32: raise NotImplementedError('Only support one divisor')
        adjusted_tensors[op.Inputs(1)] = np.array([(self.scale_factor / div_val).round().astype(np.int64)])
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
        # TODO: conditionally determine whether or not to subtract the max
        # It should depend on the input to the softmax
        if layers[-1]['layer_type'] == 'MaskNegInf':
          params = layers[-1]['params']
        elif layers[-2]['layer_type'] == 'MaskNegInf':
          params = layers[-2]['params']
          params = [params[0] - 1] + params[2:]
        else:
          params = []
      # Mean
      elif op_code == tflite.BuiltinOperator.MEAN:
        layer_type = 'Mean'
        inp_shape = interpreter.get_tensor(op.Inputs(0)).shape
        mean_idxes = interpreter.get_tensor(op.Inputs(1)).flatten().astype(np.int64)
        if len(mean_idxes) + 2 != len(inp_shape):
          raise NotImplementedError(f'Only mean over all but one axis is supported: {op_idx}')
        params = mean_idxes.tolist()
      elif op_code == tflite.BuiltinOperator.SQUARE:
        layer_type = 'Square'
        params = []
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
      elif op_code == tflite.BuiltinOperator.TANH:
        layer_type = 'Tanh'
        params = []
      elif op_code == tflite.BuiltinOperator.POW:
        layer_type = 'Pow'
        power = interpreter.get_tensor(op.Inputs(1)).flatten().astype(np.float32)
        if power != 3.: raise NotImplementedError(f'Only support power 3')
        power = power.round().astype(np.int64)
        if len(power) != 1: raise NotImplementedError(f'Only scalar power is supported: {op_idx}')
        params = power.tolist()

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
      elif op_code == tflite.BuiltinOperator.STRIDED_SLICE:
        # FIXME: this is not in general a no-op
        layer_type = 'Noop'
        params = [0]
      elif op_code == tflite.BuiltinOperator.BROADCAST_ARGS:
        layer_type = 'Noop'
        params = [0]
      elif op_code == tflite.BuiltinOperator.BROADCAST_TO:
        layer_type = 'Noop'
        params = [0]

      ## Shape
      elif op_code == tflite.BuiltinOperator.RESHAPE:
        layer_type = 'Reshape'
        params = []
      elif op_code == tflite.BuiltinOperator.TRANSPOSE:
        layer_type = 'Transpose'
        params = get_shape(interpreter, op.Inputs(0)) + interpreter.get_tensor(op.Inputs(1)).flatten().astype(np.int64).tolist()
      elif op_code == tflite.BuiltinOperator.CONCATENATION:
        # FIXME: This is not in general a no-op
        layer_type = 'Concatenation'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('Concatenation options is None')
        opt = tflite.ConcatenationOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        params = [opt.Axis()]
      elif op_code == tflite.BuiltinOperator.PACK:
        layer_type = 'Pack'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('Pack options is None')
        opt = tflite.PackOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        params = [opt.Axis()]
        if params[0] > 1: raise NotImplementedError(f'Only axis=0,1 supported at layer {op_idx}')
      elif op_code == tflite.BuiltinOperator.SPLIT:
        layer_type = 'Split'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('Split options is None')
        opt = tflite.SplitOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        axis = interpreter.get_tensor(op.Inputs(0)).flatten().astype(np.int64)[0]
        num_splits = opt.NumSplits()
        inp = interpreter.get_tensor(op.Inputs(1))
        if inp.shape[axis] % num_splits != 0:
          raise NotImplementedError(f'Only equal splits supported at layer {op_idx}')
        params = [int(axis), num_splits]
      elif op_code == tflite.BuiltinOperator.SLICE:
        layer_type = 'Slice'
        begin = interpreter.get_tensor(op.Inputs(1)).flatten().astype(np.int64).tolist()
        size = interpreter.get_tensor(op.Inputs(2)).flatten().astype(np.int64).tolist()
        params = begin + size
      elif op_code == tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
        layer_type = 'ResizeNearestNeighbor'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('ResizeNearestNeighbor options is None')
        opt = tflite.ResizeNearestNeighborOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.AlignCorners():
          raise NotImplementedError(f'Align corners not supported at layer {op_idx}')
        if not opt.HalfPixelCenters():
          raise NotImplementedError(f'Half pixel centers not supported at layer {op_idx}')
        # Can take the out shape directly from the tensor
        params = [int(opt.AlignCorners()), int(opt.HalfPixelCenters())]

      # Not implemented
      else:
        op_name = None
        for attr in dir(tflite.BuiltinOperator):
          if not attr.startswith('__'):
            if getattr(tflite.BuiltinOperator, attr) == op_code:
              op_name = attr
        raise NotImplementedError('Unsupported operator at layer {}: {}, {}'.format(op_idx, op_code, op_name))

      inp_idxes = get_inputs(op)
      # FIXME: hack for testing
      rsqrt_overflows = [99, 158, 194, 253, 289, 348]
      if op_idx in rsqrt_overflows:
        if op_code == tflite.BuiltinOperator.RSQRT:
          mask = [0, 1]
        else:
          mask = []
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

      tensor = graph.Tensors(tensor_idx)
      if tensor is None:
        raise NotImplementedError('Tensor is None')

      if tensor_idx in generated_tensor_idxes:
        print(f'skipping generated tensor: {format(tensor_idx)}, {tensor.Name()}')
        continue

      shape = []
      for i in range(tensor.ShapeLength()):
        shape.append(int(tensor.Shape(i)))
      if shape == []:
        shape = [1]

      tensor_data = interpreter.get_tensor(tensor_idx)
      if tensor.Type() == tflite.TensorType.FLOAT32:
        tensor_data = (tensor_data * self.scale_factor).round().astype(np.int64)
      elif tensor.Type() == tflite.TensorType.INT32:
        tensor_data = tensor_data.astype(np.int64)
      elif tensor.Type() == tflite.TensorType.INT64:
        continue
      else:
        raise NotImplementedError('Unsupported tensor type: {}'.format(tensor.Type()))

      if tensor_idx in adjusted_tensors:
        tensor_data = adjusted_tensors[tensor_idx]
        shape = tensor_data.shape

      tensors.append({
        'idx': tensor_idx,
        'shape': shape,
        'data': tensor_data.flatten().tolist(),
      })
      # print(tensor_idx, tensor.Type(), tensor.Name(), tensors[-1]['shape'])
      # print(np.abs(tensor_data).max())

    commit_before = []
    commit_after = []
    if self.commit:
      input_tensors = [inp['index'] for inp in input_details]
      weight_tensors = [tensor['idx'] for tensor in tensors if tensor['idx'] not in input_tensors]
      commit_before = [weight_tensors, input_tensors]

      output_tensors = [out['index'] for out in output_details]
      commit_after = [output_tensors]

    out_idxes = layers[-1]['out_idxes'] if self.expose_output else []
    d = {
      'global_sf': self.scale_factor,
      'k': self.k,
      'num_cols': self.num_cols,
      'num_random': self.num_randoms,
      'inp_idxes': [inp['index'] for inp in input_details],
      # 'out_idxes': [out['index'] for out in output_details],
      'out_idxes': out_idxes,
      'layers': layers,
      'tensors': tensors,
      'use_selectors': self.use_selectors,
      'commit_before': commit_before,
      'commit_after': commit_after,
    }
    print()
    print(d['layers'][-1])
    # d['out_idxes'] = [14]
    print(d.keys())
    print(d['out_idxes'])
    return d

  def to_msgpack(self, start_layer, end_layer, use_selectors=True):
    d = self.to_dict(start_layer, end_layer)
    model_packed = msgpack.packb(d, use_bin_type=True)
    d['tensors'] = []
    config_packed = msgpack.packb(d, use_bin_type=True)
    return model_packed, config_packed


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True)
  parser.add_argument('--model_output', type=str, required=True)
  parser.add_argument('--config_output', type=str, required=True)
  parser.add_argument('--scale_factor', type=int, default=2**16)
  parser.add_argument('--k', type=int, default=19)
  parser.add_argument('--eta', type=float, default=0.001)
  parser.add_argument('--num_cols', type=int, default=6)
  parser.add_argument('--use_selectors', action=argparse.BooleanOptionalAction, required=False, default=True)
  parser.add_argument('--commit', action=argparse.BooleanOptionalAction, required=False, default=False)
  parser.add_argument('--expose_output', action=argparse.BooleanOptionalAction, required=False, default=True)
  parser.add_argument('--start_layer', type=int, default=0)
  parser.add_argument('--end_layer', type=int, default=10000)
  parser.add_argument('--num_randoms', type=int, default=20001)
  args = parser.parse_args()

  converter = Converter(
    args.model,
    args.scale_factor,
    args.k,
    args.num_cols,
    args.num_randoms,
    args.use_selectors,
    args.commit,
    args.expose_output,
  )

  model_packed, config_packed = converter.to_msgpack(
    start_layer=args.start_layer,
    end_layer=args.end_layer,
  )
  if model_packed is None:
    raise Exception('Failed to convert model')

  with open(args.model_output, 'wb') as f:
    f.write(model_packed)
  with open(args.config_output, 'wb') as f:
    f.write(config_packed)

if __name__ == '__main__':
  main()
