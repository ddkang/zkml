use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{s, Array, IxDyn};

use crate::gadgets::{
  adder::AdderChip,
  gadget::{Gadget, GadgetConfig},
  nonlinear::exp::ExpChip,
  var_div::VarDivRoundChip,
};

use super::layer::{Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct SoftmaxChip {}

impl<F: FieldExt> Layer<F> for SoftmaxChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    let inp = &tensors[0];
    assert!(inp.ndim() == 2 || inp.ndim() == 3);

    let exp_chip = ExpChip::<F>::construct(gadget_config.clone());
    let inp_vec = inp.iter().collect::<Vec<_>>();
    let zero = constants.get(&0).unwrap().clone();
    println!("inp_vec: {:?}", inp_vec.len());
    let exped = exp_chip.forward(
      layouter.namespace(|| "exp chip"),
      &vec![inp_vec],
      &vec![zero.clone()],
    )?;
    println!("exped: {:?}", exped.len());

    let exped = Array::from_shape_vec(IxDyn(inp.shape()), exped).unwrap();

    let adder_chip = AdderChip::<F>::construct(gadget_config.clone());
    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());

    let mut outp = vec![];
    let sf = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .clone();
    if inp.ndim() == 3 {
      for i in 0..inp.shape()[0] {
        for j in 0..inp.shape()[1] {
          let exp_slice = exped.slice(s![i, j, ..]);
          let sum = adder_chip.forward(
            layouter.namespace(|| format!("sum {}", i)),
            &vec![exp_slice.iter().collect()],
            &vec![zero.clone()],
          )?;
          let sum = sum[0].clone();
          let sum_div_sf = var_div_chip.forward(
            layouter.namespace(|| format!("div {}", i)),
            &vec![vec![&sum]],
            &vec![zero.clone(), sf.clone()],
          )?;
          let sum_div_sf = sum_div_sf[0].clone();
          let dived = var_div_chip.forward(
            layouter.namespace(|| format!("div {}", i)),
            &vec![exp_slice.iter().collect()],
            &vec![zero.clone(), sum_div_sf],
          )?;
          outp.extend(dived);
        }
      }
    } else {
      panic!("Not implemented");
    }

    let outp = Array::from_shape_vec(IxDyn(inp.shape()), outp).unwrap();
    Ok(vec![outp])
  }
}
