use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::gadget::Gadget;
use crate::gadgets::{adder::AdderChip, gadget::GadgetConfig, var_div::VarDivRoundChip};

pub trait Averager<F: FieldExt> {
  fn splat<G: Clone>(&self, input: &Array<G, IxDyn>) -> Vec<Vec<G>>;

  fn get_div_val(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<AssignedCell<F, F>, Error>;

  fn avg_forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    // Due to Mean BS
    // assert_eq!(tensors.len(), 1);
    let zero = constants.get(&0).unwrap().clone();

    let inp = &tensors[0];
    let splat_inp = self.splat(inp);

    let adder_chip = AdderChip::<F>::construct(gadget_config.clone());
    let single_inputs = vec![zero.clone()];
    let mut added = vec![];
    for i in 0..splat_inp.len() {
      let tmp = splat_inp[i].iter().map(|x| x).collect::<Vec<_>>();
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
    )?;
    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());

    let single_inputs = vec![zero, div];
    let added = added.iter().map(|x| x).collect::<Vec<_>>();
    let dived = var_div_chip.forward(
      layouter.namespace(|| "average div"),
      &vec![added],
      &single_inputs,
    )?;

    Ok(dived)
  }
}
