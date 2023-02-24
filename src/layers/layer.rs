use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::gadget::GadgetConfig;

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
pub enum LayerType {
  Add,
  AvgPool2D,
  BatchMatMul,
  Conv2D,
  FullyConnected,
  Logistic,
  MaskNegInf,
  Mean,
  Mul,
  #[default]
  Noop,
  Pad,
  Reshape,
  Rsqrt,
  Softmax,
  Square,
  SquaredDifference,
  Sub,
  Transpose,
}

#[derive(Clone, Debug, Default)]
pub struct LayerConfig {
  pub layer_type: LayerType,
  pub layer_params: Vec<i64>, // This is turned into layer specific configurations at runtime
  pub inp_shapes: Vec<Vec<usize>>,
  pub out_shapes: Vec<Vec<usize>>,
  pub mask: Vec<i64>,
}

// General issue with rust: I'm not sure how to pass named arguments to a trait...
// Currently, the caller must be aware of the order of the tensors and results
pub trait Layer<F: FieldExt> {
  fn forward(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error>;
}
