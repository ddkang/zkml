use std::{convert::TryInto, marker::PhantomData};

use halo2_proofs::{
  arithmetic::FieldExt,
  circuit::{AssignedCell, Layouter},
  plonk::{Advice, Column, ConstraintSystem, Error},
};

use super::{
  poseidon::{PaddedWord, PoseidonSpongeInstructions, Sponge},
  pow5::{Pow5Chip, Pow5Config},
  primitives::{Absorbing, ConstantLength, Domain, Spec},
};

/// A Poseidon hash function, built around a sponge.
#[derive(Debug)]
pub struct MyHash<
  F: FieldExt,
  PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE>,
  S: Spec<F, T>,
  D: Domain<F, RATE>,
  const T: usize,
  const RATE: usize,
> {
  sponge: Sponge<F, PoseidonChip, S, Absorbing<PaddedWord<F>, RATE>, D, T, RATE>,
}

impl<
    F: FieldExt,
    PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE>,
    S: Spec<F, T>,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
  > MyHash<F, PoseidonChip, S, D, T, RATE>
{
  /// Initializes a new hasher.
  pub fn init(chip: PoseidonChip, layouter: impl Layouter<F>) -> Result<Self, Error> {
    Sponge::new(chip, layouter).map(|sponge| MyHash { sponge })
  }
}

impl<
    F: FieldExt,
    PoseidonChip: PoseidonSpongeInstructions<F, S, ConstantLength<L>, T, RATE>,
    S: Spec<F, T>,
    const T: usize,
    const RATE: usize,
    const L: usize,
  > MyHash<F, PoseidonChip, S, ConstantLength<L>, T, RATE>
{
  /// Hashes the given input.
  pub fn hash(
    mut self,
    mut layouter: impl Layouter<F>,
    message: Vec<AssignedCell<F, F>>,
  ) -> Result<AssignedCell<F, F>, Error> {
    assert!(
      message.len() % L == 0,
      "message length must be a multiple of L"
    );
    for (i, value) in message
      .into_iter()
      .map(PaddedWord::Message)
      .chain(<ConstantLength<L> as Domain<F, RATE>>::padding(L).map(PaddedWord::Padding))
      .enumerate()
    {
      self
        .sponge
        .absorb(layouter.namespace(|| format!("absorb_{}", i)), value)?;
    }
    self
      .sponge
      .finish_absorbing(layouter.namespace(|| "finish absorbing"))?
      .squeeze(layouter.namespace(|| "squeeze"))
  }
}

#[derive(Clone, Debug)]
pub struct HasherConfig<F: FieldExt, const WIDTH: usize, const RATE: usize, const L: usize> {
  poseidon_config: Pow5Config<F, WIDTH, RATE>,
}

pub struct HasherChip<F: FieldExt, const WIDTH: usize, const RATE: usize, const L: usize> {
  config: HasherConfig<F, WIDTH, RATE, L>,
  _marker: PhantomData<F>,
}

impl<F: FieldExt, const WIDTH: usize, const RATE: usize, const L: usize>
  HasherChip<F, WIDTH, RATE, L>
{
  pub fn construct(config: HasherConfig<F, WIDTH, RATE, L>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure<S: Spec<F, WIDTH>>(
    meta: &mut ConstraintSystem<F>,
    state: [Column<Advice>; WIDTH],
    partial_sbox: Column<Advice>,
  ) -> HasherConfig<F, WIDTH, RATE, L> {
    let rc_a = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();
    let rc_b = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();

    meta.enable_constant(rc_b[0]);

    HasherConfig {
      poseidon_config: Pow5Chip::configure::<S>(
        meta,
        state.try_into().unwrap(),
        partial_sbox,
        rc_a.try_into().unwrap(),
        rc_b.try_into().unwrap(),
      ),
    }
  }

  pub fn hash<S: Spec<F, WIDTH>>(
    &self,
    layouter: &mut impl Layouter<F>,
    weights: &Vec<AssignedCell<F, F>>,
  ) -> Result<AssignedCell<F, F>, Error> {
    let chip = Pow5Chip::construct(self.config.poseidon_config.clone());
    let hasher =
      MyHash::<F, _, S, ConstantLength<L>, WIDTH, RATE>::init(chip, layouter.namespace(|| ""))
        .unwrap();

    let hash = hasher
      .hash(layouter.namespace(|| ""), weights.to_vec())
      .unwrap();
    Ok(hash)
  }
}
