use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::gadget::GadgetConfig;

use super::super::layer::{Layer, LayerConfig};

pub struct ReshapeChip {}

impl<F: FieldExt> Layer<F> for ReshapeChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    _constants: &HashMap<i64, AssignedCell<F, F>>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    let inp = &tensors[0];
    let shape = layer_config.out_shapes[0].clone();

    println!("Reshape: {:?} -> {:?}", inp.shape(), shape);
    let flat = inp.iter().map(|x| x.clone()).collect();
    let out = Array::from_shape_vec(shape, flat).unwrap();
    Ok(vec![out])
  }
}
