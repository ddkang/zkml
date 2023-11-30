use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::ff::PrimeField,
  plonk::Error,
};
use ndarray::{s, Array, IxDyn};

use crate::gadgets::{
  adder::AdderChip,
  gadget::{Gadget, GadgetConfig, GadgetType},
  max::MaxChip,
  nonlinear::exp::ExpGadgetChip,
  sub_pairs::SubPairsChip,
  var_div_big3::VarDivRoundBig3Chip,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct SoftmaxChip {}

impl SoftmaxChip {
  pub fn softmax_flat<F: PrimeField>(
    mut layouter: impl Layouter<F>,
    constants: &HashMap<i64, CellRc<F>>,
    inp_flat: Vec<(&AssignedCell<F, F>, F)>,
    gadget_config: Rc<GadgetConfig>,
    mask: &Vec<i64>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let exp_chip = ExpGadgetChip::<F>::construct(gadget_config.clone());
    let adder_chip = AdderChip::<F>::construct(gadget_config.clone());
    let sub_pairs_chip = SubPairsChip::<F>::construct(gadget_config.clone());
    let max_chip = MaxChip::<F>::construct(gadget_config.clone());
    let var_div_big_chip = VarDivRoundBig3Chip::<F>::construct(gadget_config.clone());

    let zero = constants.get(&0).unwrap().as_ref();
    let sf_cell = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .as_ref();
    let sf = {
      let shift_val_i64 = -gadget_config.min_val * 2;
      let shift_val_f = F::from(shift_val_i64 as u64);
      F::from((gadget_config.scale_factor as i64 + shift_val_i64) as u64) - shift_val_f
    };

    // Mask the input for max computation and subtraction
    let inp_take = inp_flat
      .iter()
      .enumerate()
      .filter(|(i, _)| mask[*i] == 0) // Awkwardly, 1 = take negative infinity
      .map(|(_, x)| *x)
      .collect::<Vec<_>>();

    // Compute the max
    let max = max_chip
      .forward(
        layouter.namespace(|| format!("max")),
        &vec![inp_take.clone()],
        &vec![(zero, F::ZERO)],
      )
      .unwrap();
    let max = (&max[0].0, max[0].1);

    // Subtract the max
    let max_flat = vec![max; inp_take.len()];
    let sub = sub_pairs_chip.forward(
      layouter.namespace(|| format!("sub")),
      &vec![inp_take, max_flat],
      &vec![(zero, F::ZERO)],
    )?;

    let sub = sub.iter().map(|x| (&x.0, x.1)).collect::<Vec<_>>();

    // Compute the exp
    let exp_slice = exp_chip.forward(
      layouter.namespace(|| format!("exp")),
      &vec![sub],
      &vec![(zero, F::ZERO)],
    )?;

    // Compute the sum
    let sum = adder_chip.forward(
      layouter.namespace(|| format!("sum")),
      &vec![exp_slice.iter().map(|x| (&x.0, x.1)).collect()],
      &vec![(zero, F::ZERO)],
    )?;
    let sum = (&sum[0].0, sum[0].1);
    let sum_div_sf = var_div_big_chip.forward(
      layouter.namespace(|| format!("sum div sf")),
      &vec![vec![sum]],
      &vec![
        (zero, F::ZERO), 
        (sf_cell, sf)
      ],
    )?;
    let sum_div_sf = (&sum_div_sf[0].0, sum_div_sf[0].1);

    let dived = var_div_big_chip.forward(
      layouter.namespace(|| format!("div")),
      &vec![exp_slice.iter().map(|x| (&x.0, x.1)).collect()],
      &vec![(zero, F::ZERO), sum_div_sf],
    )?;

    // Take either zero (softmax(-inf)) or the result
    let mut div_idx = 0;
    let dived = mask
      .iter()
      .map(|x| {
        if *x == 1 {
          (zero.clone(), F::ZERO)
        } else {
          let tmp = dived[div_idx].clone();
          div_idx = div_idx + 1;
          tmp
        }
      })
      .collect();

    Ok(dived)
  }
}

impl<F: PrimeField> Layer<F> for SoftmaxChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
    assert!(inp.ndim() == 2 || inp.ndim() == 3 || inp.ndim() == 4);
    if inp.ndim() == 4 {
      assert_eq!(inp.shape()[0], 1);
    }

    let inp_shape = inp.shape().iter().map(|x| *x).collect::<Vec<_>>();
    let mask = if layer_config.layer_params.len() == 0 {
      Array::from_shape_fn(IxDyn(&inp_shape), |_| 0)
    } else {
      let mask_shape_len = layer_config.layer_params[0] as usize;
      let mask_shape = layer_config.layer_params[1..(1 + mask_shape_len)]
        .iter()
        .map(|x| *x as usize)
        .collect::<Vec<_>>();
      let mask = layer_config.layer_params[(1 + mask_shape_len)..].to_vec();
      let mask = Array::from_shape_vec(IxDyn(&mask_shape), mask).unwrap();
      let mask = mask.broadcast(IxDyn(&inp_shape)).unwrap().to_owned();
      mask
    };

    let shape = if inp.ndim() == 2 || inp.ndim() == 3 {
      inp.shape().iter().map(|x| *x).collect::<Vec<_>>()
    } else {
      vec![inp.shape()[1], inp.shape()[2], inp.shape()[3]]
    };
    let inp = inp.to_owned().into_shape(shape.clone()).unwrap();
    let mask = mask.into_shape(shape.clone()).unwrap();

    let mut outp = vec![];
    if inp.ndim() == 2 {
      for i in 0..shape[0] {
        let inp_slice = inp.slice(s![i, ..]);
        let inp_flat = inp_slice.iter().map(|x| (x.0.as_ref(), x.1)).collect::<Vec<_>>();
        let mask_slice = mask.slice(s![i, ..]);
        let mask_flat = mask_slice.iter().map(|x| *x as i64).collect::<Vec<_>>();
        let dived = Self::softmax_flat(
          layouter.namespace(|| format!("softmax {}", i)),
          constants,
          inp_flat,
          gadget_config.clone(),
          &mask_flat,
        )
        .unwrap();
        outp.extend(dived);
      }
    } else if inp.ndim() == 3 {
      for i in 0..shape[0] {
        for j in 0..shape[1] {
          let inp_slice = inp.slice(s![i, j, ..]);
          let inp_flat = inp_slice.iter().map(|x| (x.0.as_ref(), x.1)).collect::<Vec<_>>();
          let mask_slice = mask.slice(s![i, j, ..]);
          let mask_flat = mask_slice.iter().map(|x| *x as i64).collect::<Vec<_>>();
          let dived = Self::softmax_flat(
            layouter.namespace(|| format!("softmax {} {}", i, j)),
            constants,
            inp_flat,
            gadget_config.clone(),
            &mask_flat,
          )
          .unwrap();
          outp.extend(dived);
        }
      }
    } else {
      panic!("Not implemented");
    }

    let outp = outp.into_iter().map(|x| (Rc::new(x.0), x.1)).collect::<Vec<_>>();
    let outp = Array::from_shape_vec(IxDyn(inp.shape()), outp).unwrap();
    Ok(vec![outp])
  }
}

impl GadgetConsumer for SoftmaxChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::Exp,
      GadgetType::Adder,
      GadgetType::VarDivRoundBig3,
      GadgetType::Max,
      GadgetType::SubPairs,
      GadgetType::InputLookup,
    ]
  }
}
