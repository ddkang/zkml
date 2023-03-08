use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{s, Array, IxDyn};

use crate::gadgets::{
  adder::AdderChip,
  gadget::{Gadget, GadgetConfig, GadgetType},
  max::MaxChip,
  nonlinear::exp::ExpGadgetChip,
  sub_pairs::SubPairsChip,
  var_div_big::VarDivRoundBigChip,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct SoftmaxChip {}

impl SoftmaxChip {
  pub fn softmax_flat<F: FieldExt>(
    mut layouter: impl Layouter<F>,
    constants: &HashMap<i64, CellRc<F>>,
    inp_flat: Vec<&AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let exp_chip = ExpGadgetChip::<F>::construct(gadget_config.clone());
    let adder_chip = AdderChip::<F>::construct(gadget_config.clone());
    let sub_pairs_chip = SubPairsChip::<F>::construct(gadget_config.clone());
    let max_chip = MaxChip::<F>::construct(gadget_config.clone());
    let var_div_big_chip = VarDivRoundBigChip::<F>::construct(gadget_config.clone());

    let zero = constants.get(&0).unwrap().as_ref();
    let sf = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .as_ref();

    // Compute the max
    let max = max_chip
      .forward(
        layouter.namespace(|| format!("max")),
        &vec![inp_flat.clone()],
        &vec![zero],
      )
      .unwrap();
    let max = &max[0];

    // Subtract the max
    let max_flat = vec![max; inp_flat.len()];
    let sub = sub_pairs_chip.forward(
      layouter.namespace(|| format!("sub")),
      &vec![inp_flat, max_flat],
      &vec![zero],
    )?;
    let sub = sub.iter().collect::<Vec<_>>();

    // Compute the exp
    let exp_slice = exp_chip.forward(
      layouter.namespace(|| format!("exp")),
      &vec![sub],
      &vec![zero],
    )?;

    // Compute the sum
    let sum = adder_chip.forward(
      layouter.namespace(|| format!("sum")),
      &vec![exp_slice.iter().collect()],
      &vec![zero],
    )?;
    let sum = sum[0].clone();
    let sum_div_sf = var_div_big_chip.forward(
      layouter.namespace(|| format!("sum div sf")),
      &vec![vec![&sum]],
      &vec![zero, sf],
    )?;
    let sum_div_sf = sum_div_sf[0].clone();

    let dived = var_div_big_chip.forward(
      layouter.namespace(|| format!("div")),
      &vec![exp_slice.iter().collect()],
      &vec![zero, &sum_div_sf],
    )?;

    Ok(dived)
  }
}

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
    assert!(inp.ndim() == 2 || inp.ndim() == 3 || inp.ndim() == 4);
    if inp.ndim() == 4 {
      assert_eq!(inp.shape()[0], 1);
    }

    let shape = if inp.ndim() == 2 || inp.ndim() == 3 {
      inp.shape().iter().map(|x| *x).collect::<Vec<_>>()
    } else {
      vec![inp.shape()[1], inp.shape()[2], inp.shape()[3]]
    };
    let inp = inp.to_owned().into_shape(shape.clone()).unwrap();

    let mut outp = vec![];
    if inp.ndim() == 3 {
      for i in 0..shape[0] {
        for j in 0..shape[1] {
          let inp_slice = inp.slice(s![i, j, ..]);
          let inp_flat = inp_slice.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
          let dived = Self::softmax_flat(
            layouter.namespace(|| format!("softmax {} {}", i, j)),
            constants,
            inp_flat,
            gadget_config.clone(),
          )
          .unwrap();
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
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::Exp,
      GadgetType::Adder,
      GadgetType::VarDivRoundBig,
      GadgetType::Max,
      GadgetType::SubPairs,
      GadgetType::InputLookup,
    ]
  }
}
