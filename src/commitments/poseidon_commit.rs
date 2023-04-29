use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_gadgets::poseidon::{
  primitives::{generate_constants, Absorbing, ConstantLength, Domain, Mds, Spec},
  PaddedWord, PoseidonSpongeInstructions, Pow5Chip, Pow5Config, Sponge,
};
use halo2_proofs::{
  circuit::Layouter,
  halo2curves::ff::{FromUniformBytes, PrimeField},
  plonk::{Advice, Column, ConstraintSystem, Error},
};

use crate::{gadgets::gadget::GadgetConfig, layers::layer::CellRc};

use super::commit::Commit;

pub const WIDTH: usize = 3;
pub const RATE: usize = 2;
pub const L: usize = 8 - WIDTH - 1;

#[derive(Clone, Debug)]

pub struct PoseidonCommitChip<
  F: PrimeField + Ord + FromUniformBytes<64>,
  const WIDTH: usize,
  const RATE: usize,
  const L: usize,
> {
  pub poseidon_config: Pow5Config<F, WIDTH, RATE>,
}

#[derive(Debug)]
pub struct P128Pow5T3Gen<F: PrimeField, const SECURE_MDS: usize>(PhantomData<F>);

impl<F: PrimeField, const SECURE_MDS: usize> P128Pow5T3Gen<F, SECURE_MDS> {
  pub fn new() -> Self {
    P128Pow5T3Gen(PhantomData::default())
  }
}

impl<F: FromUniformBytes<64> + Ord, const SECURE_MDS: usize> Spec<F, 3, 2>
  for P128Pow5T3Gen<F, SECURE_MDS>
{
  fn full_rounds() -> usize {
    8
  }

  fn partial_rounds() -> usize {
    56
  }

  fn sbox(val: F) -> F {
    val.pow_vartime([5])
  }

  fn secure_mds() -> usize {
    SECURE_MDS
  }

  fn constants() -> (Vec<[F; 3]>, Mds<F, 3>, Mds<F, 3>) {
    generate_constants::<_, Self, 3, 2>()
  }
}

/// A Poseidon hash function, built around a sponge.
#[derive(Debug)]
pub struct MyHash<
  F: PrimeField,
  PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE>,
  S: Spec<F, T, RATE>,
  D: Domain<F, RATE>,
  const T: usize,
  const RATE: usize,
> {
  pub sponge: Sponge<F, PoseidonChip, S, Absorbing<PaddedWord<F>, RATE>, D, T, RATE>,
}

impl<F: PrimeField + Ord + FromUniformBytes<64>> PoseidonCommitChip<F, WIDTH, RATE, L> {
  pub fn configure(
    meta: &mut ConstraintSystem<F>,
    // TODO: ??
    _input: [Column<Advice>; L],
    state: [Column<Advice>; WIDTH],
    partial_sbox: Column<Advice>,
  ) -> PoseidonCommitChip<F, WIDTH, RATE, L> {
    let rc_a = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();
    let rc_b = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();

    meta.enable_constant(rc_b[0]);

    PoseidonCommitChip {
      poseidon_config: Pow5Chip::configure::<P128Pow5T3Gen<F, 0>>(
        meta,
        state.try_into().unwrap(),
        partial_sbox,
        rc_a.try_into().unwrap(),
        rc_b.try_into().unwrap(),
      ),
    }
  }
}

impl<F: PrimeField + Ord + FromUniformBytes<64>> Commit<F>
  for PoseidonCommitChip<F, WIDTH, RATE, L>
{
  fn commit(
    &self,
    mut layouter: impl Layouter<F>,
    _gadget_config: Rc<GadgetConfig>,
    _constants: &HashMap<i64, CellRc<F>>,
    values: &Vec<CellRc<F>>,
    blinding: CellRc<F>,
  ) -> Result<Vec<CellRc<F>>, Error> {
    let chip = Pow5Chip::construct(self.poseidon_config.clone());
    let mut hasher: MyHash<F, Pow5Chip<F, 3, 2>, P128Pow5T3Gen<F, 0>, ConstantLength<L>, 3, 2> =
      Sponge::new(chip, layouter.namespace(|| "sponge"))
        .map(|sponge| MyHash { sponge })
        .unwrap();

    let mut new_vals = values
      .iter()
      .map(|x| x.clone())
      .chain(vec![blinding.clone()])
      .collect::<Vec<_>>();
    while new_vals.len() % L != 0 {
      new_vals.push(blinding.clone());
    }
    for (i, value) in new_vals
      .iter()
      .map(|x| PaddedWord::Message((**x).clone()))
      .chain(<ConstantLength<L> as Domain<F, RATE>>::padding(L).map(PaddedWord::Padding))
      .enumerate()
    {
      hasher
        .sponge
        .absorb(layouter.namespace(|| format!("absorb {}", i)), value)
        .unwrap();
    }
    let outp = hasher
      .sponge
      .finish_absorbing(layouter.namespace(|| "finish absorbing"))
      .unwrap()
      .squeeze(layouter.namespace(|| "squeeze"))
      .unwrap();
    let outp = Rc::new(outp);

    Ok(vec![outp])
  }
}
