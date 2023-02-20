use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::gadget::GadgetConfig;

use super::layer::{Layer, LayerConfig};

pub struct NoopChip {}

impl<F: FieldExt> Layer<F> for NoopChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    _constants: &HashMap<i64, AssignedCell<F, F>>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    let ret_idx = layer_config.layer_params[0] as usize;
    Ok(vec![tensors[ret_idx].clone()])
  }
}
