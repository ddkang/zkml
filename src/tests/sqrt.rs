use std::{
  collections::{HashMap, HashSet},
  marker::PhantomData,
};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  dev::MockProver,
  halo2curves::{bn256::Fr, FieldExt},
};
use ndarray::{Array, ArrayBase, Dim, IxDyn, IxDynImpl, OwnedRepr};

use crate::gadgets::gadget::Gadget;
use crate::{
  gadgets::{
    gadget::{self, GadgetConfig, GadgetType},
    sqrt_big::SqrtBigChip,
  },
  tests::test_circuit::{TestCircuit, K},
  utils::loader::TensorMsgpack,
};

use super::test_circuit::{TestCircFunc, TestConfig};

pub struct TestFunc {}

impl<F: FieldExt> TestCircFunc<F> for TestFunc {
  fn compute_tensor(
    layouter: &mut impl Layouter<F>,
    config: &TestConfig<F>,
    tensors: &Vec<ArrayBase<OwnedRepr<AssignedCell<F, F>>, Dim<IxDynImpl>>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
  ) {
    let inp = &tensors[0];
    let inp_vec = inp.iter().collect::<Vec<_>>();

    let zero = constants.get(&0).unwrap().clone();

    let sqrt_chip = SqrtBigChip::<F>::construct(config.gadget_config.clone().into());
    let _result = sqrt_chip.forward(layouter.namespace(|| "test"), &vec![inp_vec], &vec![zero]);
  }
}

#[test]
fn test_sqrt() {
  let input_tensor = TensorMsgpack {
    idx: 0,
    shape: vec![3],
    data: vec![5, 10, 100],
  };
  let input_tensors: Vec<TensorMsgpack> = vec![input_tensor];

  let gadget = &crate::tests::test_circuit::GADGET_CONFIG;
  let cloned_gadget = gadget.lock().unwrap().clone();

  let circuit = TestCircuit::<Fr, TestFunc>::new(input_tensors);

  let outp = vec![];
  let prover = MockProver::run(K.try_into().unwrap(), &circuit, vec![outp.clone()]).unwrap();
  assert_eq!(prover.verify(), Ok(()));
}
