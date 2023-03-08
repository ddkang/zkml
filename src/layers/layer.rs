use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::gadget::{GadgetConfig, GadgetType};

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
pub enum LayerType {
  Add,
  AvgPool2D,
  BatchMatMul,
  Concatenation,
  Conv2D,
  FullyConnected,
  Logistic,
  MaskNegInf,
  MaxPool2D,
  Mean,
  Mul,
  #[default]
  Noop,
  Pack,
  Pad,
  Reshape,
  Rsqrt,
  Slice,
  Softmax,
  Split,
  Square,
  SquaredDifference,
  Sub,
  Transpose,
}

// NOTE: This is the same order as the TFLite schema
// Must not be changed
#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum ActivationType {
  #[default]
  None,
  Relu,
  ReluN1To1,
  Relu6,
  Tanh,
  SignBit,
}

#[derive(Clone, Debug, Default)]
pub struct LayerConfig {
  pub layer_type: LayerType,
  pub layer_params: Vec<i64>, // This is turned into layer specific configurations at runtime
  pub inp_shapes: Vec<Vec<usize>>,
  pub out_shapes: Vec<Vec<usize>>,
  pub mask: Vec<i64>,
}

pub type CellRc<F> = Rc<AssignedCell<F, F>>;
pub type AssignedTensor<F> = Array<CellRc<F>, IxDyn>;
// General issue with rust: I'm not sure how to pass named arguments to a trait...
// Currently, the caller must be aware of the order of the tensors and results
pub trait Layer<F: FieldExt> {
  fn forward(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error>;
}

pub trait GadgetConsumer {
  fn used_gadgets(&self, layer_params: Vec<i64>) -> Vec<GadgetType>;
}
