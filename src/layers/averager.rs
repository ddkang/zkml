use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::ff::PrimeField,
  plonk::Error,
};

use crate::gadgets::gadget::Gadget;
use crate::gadgets::{adder::AdderChip, gadget::GadgetConfig, var_div::VarDivRoundChip};

use super::layer::{AssignedTensor, CellRc, LayerConfig};

pub trait Averager<F: PrimeField> {
  fn splat(&self, input: &AssignedTensor<F>, layer_config: &LayerConfig) -> Vec<Vec<(CellRc<F>, F)>>;

  fn get_div_val(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<(AssignedCell<F, F>, F), Error>;

  fn avg_forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<(CellRc<F>, F)>, Error> {
    // Due to Mean BS
    // assert_eq!(tensors.len(), 1);
    let zero = constants.get(&0).unwrap().as_ref();

    let inp = &tensors[0];
    let splat_inp = self.splat(inp, layer_config);

    let adder_chip = AdderChip::<F>::construct(gadget_config.clone());
    let single_inputs = vec![(zero, F::ZERO)];
    let mut added = vec![];
    for i in 0..splat_inp.len() {
      let tmp = splat_inp[i].iter().map(|x| (x.0.as_ref(), x.1)).collect::<Vec<_>>();
      let tmp = adder_chip.forward(
        layouter.namespace(|| format!("average {}", i)),
        &vec![tmp],
        &single_inputs,
      )?;
      added.push(tmp[0].clone());
    }

    let div = self.get_div_val(
      layouter.namespace(|| "average div"),
      tensors,
      gadget_config.clone(),
      layer_config,
    )?;
    let div = (&div.0, div.1);

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());

    let single_inputs = vec![(zero, F::ZERO), div];
    let added = added.iter().map(|x| (&x.0, x.1)).collect::<Vec<_>>();
    let dived = var_div_chip.forward(
      layouter.namespace(|| "average div"),
      &vec![added],
      &single_inputs,
    )?;
    let dived = dived.into_iter().map(|x| (Rc::new(x.0), x.1)).collect::<Vec<_>>();

    Ok(dived)
  }
}
