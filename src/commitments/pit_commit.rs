use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};

use crate::{
  gadgets::{
    dot_prod::DotProductChip,
    gadget::{Gadget, GadgetConfig},
  },
  layers::layer::CellRc,
  utils::helpers::{NUM_RANDOMS, RAND_START_IDX},
};

use super::commit::Commit;

pub struct PITCommitChip<F: PrimeField> {
  pub marker: PhantomData<F>,
}

impl<F: PrimeField> Commit<F> for PITCommitChip<F> {
  fn commit(
    &self,
    mut layouter: impl Layouter<F>,
    gadget_config: Rc<GadgetConfig>,
    constants: &HashMap<i64, CellRc<F>>,
    values: &Vec<CellRc<F>>,
    blinding: CellRc<F>,
  ) -> Result<Vec<CellRc<F>>, Error> {
    let dot_prod_chip = DotProductChip::<F>::construct(gadget_config.clone());
    let dot_prod_size = NUM_RANDOMS;

    let mut rands = vec![];
    for i in 0..dot_prod_size {
      let tmp = RAND_START_IDX + i;
      rands.push(constants.get(&tmp).unwrap().as_ref());
    }

    let zero = constants.get(&0).unwrap().clone();

    let mut outp = vec![];
    // -1 for blinding
    for chunk in values.chunks((dot_prod_size - 1).try_into().unwrap()) {
      let mut inp = chunk.to_vec();
      inp.push(blinding.clone());
      let inp = inp.iter().map(|x| x.as_ref()).collect();

      let vec_inp = vec![inp, rands.clone()];
      let single_inp = vec![zero.as_ref()];
      let res = dot_prod_chip.forward(layouter.namespace(|| "dot_prod"), &vec_inp, &single_inp)?;
      outp.push(Rc::new(res[0].clone()));
    }

    Ok(outp)
  }
}
