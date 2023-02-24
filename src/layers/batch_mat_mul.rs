use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::FieldExt, plonk::Error};
use ndarray::{Array, Axis, IxDyn};

use crate::gadgets::gadget::GadgetConfig;

use super::{
  fully_connected::FullyConnectedChip,
  layer::{AssignedTensor, CellRc, Layer, LayerConfig},
};

pub struct BatchMatMulChip {}

impl<F: FieldExt> Layer<F> for BatchMatMulChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp1 = &tensors[0];
    let inp2 = &tensors[1];
    println!("inp1: {:?}", inp1.shape());
    println!("inp2: {:?}", inp2.shape());

    assert_eq!(inp1.ndim(), 3);
    assert_eq!(inp2.ndim(), 3);
    assert_eq!(inp1.shape()[0], inp2.shape()[0]);
    assert_eq!(inp1.shape()[2], inp2.shape()[1]);

    let out_shape = vec![inp1.shape()[0], inp1.shape()[1], inp2.shape()[2]];

    let fc_chip = FullyConnectedChip::<F> {
      _marker: PhantomData,
    };

    let mut outp: Vec<CellRc<F>> = vec![];
    for i in 0..inp1.shape()[0] {
      let inp1_slice = inp1.index_axis(Axis(0), i).to_owned();
      // Due to tensorflow BS, transpose the "weights"
      let inp2_slice = inp2.index_axis(Axis(0), i).t().to_owned();
      println!("inp1_slice: {:?}", inp1_slice.shape());
      println!("inp2_slice: {:?}", inp2_slice.shape());
      let outp_slice = fc_chip.forward(
        layouter.namespace(|| ""),
        &vec![inp1_slice, inp2_slice],
        constants,
        gadget_config.clone(),
        layer_config,
      )?;
      outp.extend(outp_slice[0].iter().map(|x| x.clone()).collect::<Vec<_>>());
    }

    let outp = Array::from_shape_vec(IxDyn(out_shape.as_slice()), outp).unwrap();
    Ok(vec![outp])
  }
}
