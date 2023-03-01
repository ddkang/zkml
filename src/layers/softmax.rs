use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{circuit::Layouter, halo2curves::FieldExt, plonk::Error};
use ndarray::{s, Array, IxDyn};

use crate::gadgets::{
  adder::AdderChip,
  gadget::{Gadget, GadgetConfig, GadgetType},
  nonlinear::exp::ExpGadgetChip,
  sqrt_big::SqrtBigChip,
  var_div::VarDivRoundChip,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct SoftmaxChip {}

impl<F: FieldExt> Layer<F> for SoftmaxChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
    assert!(inp.ndim() == 2 || inp.ndim() == 3);

    let exp_chip = ExpGadgetChip::<F>::construct(gadget_config.clone());
    let inp_vec = inp.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let zero = constants.get(&0).unwrap().as_ref();
    println!("inp_vec: {:?}", inp_vec.len());
    let exped = exp_chip.forward(
      layouter.namespace(|| "exp chip"),
      &vec![inp_vec],
      &vec![zero],
    )?;
    println!("exped: {:?}", exped.len());

    let exped = Array::from_shape_vec(IxDyn(inp.shape()), exped).unwrap();

    let adder_chip = AdderChip::<F>::construct(gadget_config.clone());
    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());

    let mut outp = vec![];
    let sf = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .as_ref();
    if inp.ndim() == 3 {
      for i in 0..inp.shape()[0] {
        for j in 0..inp.shape()[1] {
          // Compute the sum
          let exp_slice = exped.slice(s![i, j, ..]);
          let sum = adder_chip.forward(
            layouter.namespace(|| format!("sum {}", i)),
            &vec![exp_slice.iter().collect()],
            &vec![zero],
          )?;
          let sum = sum[0].clone();

          // Divide by the scale factor
          let sum_div_sf = var_div_chip.forward(
            layouter.namespace(|| format!("div {}", i)),
            &vec![vec![&sum]],
            &vec![zero, sf],
          )?;
          let sum_div_sf = sum_div_sf[0].clone();

          // Compute the sqrt of the sum div SF
          let sqrt_big_chip = SqrtBigChip::<F>::construct(gadget_config.clone());
          let sqrt = sqrt_big_chip.forward(
            layouter.namespace(|| format!("sqrt {}", i)),
            &vec![vec![&sum_div_sf]],
            &vec![zero],
          )?;
          let sqrt = &sqrt[0];

          let dived = var_div_chip.forward(
            layouter.namespace(|| format!("div {}", i)),
            &vec![exp_slice.iter().collect()],
            &vec![zero, sqrt],
          )?;
          let dived = dived.iter().collect::<Vec<_>>();
          let dived = var_div_chip.forward(
            layouter.namespace(|| format!("div {}", i)),
            &vec![dived],
            &vec![zero, sqrt],
          )?;
          outp.extend(dived);
        }
      }
    } else {
      panic!("Not implemented");
    }

    let outp = outp.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>();
    let outp = Array::from_shape_vec(IxDyn(inp.shape()), outp).unwrap();
    Ok(vec![outp])
  }
}

impl GadgetConsumer for SoftmaxChip {
  fn used_gadgets(&self) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::Exp,
      GadgetType::Adder,
      GadgetType::VarDivRound,
      GadgetType::SqrtBig,
    ]
  }
}
