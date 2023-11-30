// TODO: Speed up Depthwise operations with Freivald's algorithm

use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::ff::PrimeField,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    bias_div_round_relu6::BiasDivRoundRelu6Chip,
    dot_prod::DotProductChip,
    gadget::{Gadget, GadgetConfig, GadgetType},
    nonlinear::relu::ReluChip,
  },
  layers::{
    fully_connected::{FullyConnectedChip, FullyConnectedConfig},
    shape::pad::pad,
  },
};

use super::layer::{ActivationType, AssignedTensor, GadgetConsumer, Layer, LayerConfig, CellRc};

#[derive(Default, Clone, Copy, Eq, PartialEq)]
pub enum PaddingEnum {
  #[default]
  Same,
  Valid,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConvLayerEnum {
  #[default]
  Conv2D,
  DepthwiseConv2D,
}

pub struct Conv2DConfig {
  pub conv_type: ConvLayerEnum,
  pub padding: PaddingEnum,
  pub activation: ActivationType,
  pub stride: (usize, usize),
}

pub struct Conv2DChip<F: PrimeField> {
  pub config: LayerConfig,
  pub _marker: PhantomData<F>,
}

impl<F: PrimeField> Conv2DChip<F> {
  // TODO: this is horrible. What's the best way to fix this?
  pub fn param_vec_to_config(layer_params: Vec<i64>) -> Conv2DConfig {
    let conv_type = match layer_params[0] {
      0 => ConvLayerEnum::Conv2D,
      1 => ConvLayerEnum::DepthwiseConv2D,
      _ => panic!("Invalid conv type"),
    };
    let padding = match layer_params[1] {
      0 => PaddingEnum::Same,
      1 => PaddingEnum::Valid,
      _ => panic!("Invalid padding"),
    };
    let activation = match layer_params[2] {
      0 => ActivationType::None,
      1 => ActivationType::Relu,
      3 => ActivationType::Relu6,
      _ => panic!("Invalid activation type"),
    };
    let stride = (layer_params[3] as usize, layer_params[4] as usize);
    Conv2DConfig {
      conv_type,
      padding,
      activation,
      stride,
    }
  }

  pub fn get_padding(
    h: usize,
    w: usize,
    si: usize,
    sj: usize,
    ci: usize,
    cj: usize,
  ) -> ((usize, usize), (usize, usize)) {
    let ph = if h % si == 0 {
      (ci as i64 - sj as i64).max(0)
    } else {
      (ci as i64 - (h % si) as i64).max(0)
    } as usize;
    let pw = if w % sj == 0 {
      (cj as i64 - sj as i64).max(0)
    } else {
      (cj as i64 - (w % sj) as i64).max(0)
    } as usize;
    ((ph / 2, ph - ph / 2), (pw / 2, pw - pw / 2))
  }

  pub fn out_hw(
    h: usize,
    w: usize,
    si: usize,
    sj: usize,
    ch: usize,
    cw: usize,
    padding: PaddingEnum,
  ) -> (usize, usize) {
    /*
    println!(
      "H: {}, W: {}, SI: {}, SJ: {}, CH: {}, CW: {}",
      h, w, si, sj, ch, cw
    );
    */
    // https://iq.opengenus.org/same-and-valid-padding/
    match padding {
      PaddingEnum::Same => ((h + si - 1) / si, (w + sj - 1) / sj),
      // TODO: the above is probably correct, but we always have valid paddings
      // PaddingEnum::Same => (h / si, w / sj),
      PaddingEnum::Valid => ((h - ch) / si + 1, (w - cw) / sj + 1),
    }
  }

  pub fn splat<G: Clone>(
    &self,
    tensors: &Vec<Array<(Rc<G>, F), IxDyn>>,
    zero: Rc<G>,
  ) -> (Vec<Vec<(Rc<G>, F)>>, Vec<Vec<(Rc<G>, F)>>, Vec<(Rc<G>, F)>) {
    // assert_eq!(tensors.len(), 3);
    assert!(tensors.len() <= 3);

    let conv_config = &Self::param_vec_to_config(self.config.layer_params.clone());

    let inp = &tensors[0];
    let weights = &tensors[1];
    let zero_arr = Array::from_elem(IxDyn(&vec![1]), (zero.clone(), F::ZERO));
    let biases = if tensors.len() == 3 {
      &tensors[2]
    } else {
      &zero_arr
    };

    let h: usize = inp.shape()[1];
    let w: usize = inp.shape()[2];

    let ch: usize = weights.shape()[1];
    let cw: usize = weights.shape()[2];

    let (si, sj) = conv_config.stride;

    // B, H, W, C
    assert_eq!(inp.shape().len(), 4);

    let (ph, pw) = if conv_config.padding == PaddingEnum::Same {
      Self::get_padding(h, w, si, sj, ch, cw)
    } else {
      ((0, 0), (0, 0))
    };
    // println!("Padding: {:?}", (ph, pw));
    let padding = vec![[0, 0], [ph.0, ph.1], [pw.0, pw.1], [0, 0]];

    let inp_pad = pad(&inp, padding, &(zero.clone(), F::ZERO));

    let (oh, ow) = Self::out_hw(h, w, si, sj, ch, cw, conv_config.padding);

    let mut inp_cells = vec![];
    let mut weights_cells = vec![];
    let mut biases_cells = vec![];
    let mut input_row_idx = 0;
    let mut weight_row_idx = 0;

    // (output_channels x inp_channels * C_H * C_W)
    for chan_out in 0..weights.shape()[0] {
      weights_cells.push(vec![]);
      for ci in 0..weights.shape()[1] {
        for cj in 0..weights.shape()[2] {
          for ck in 0..weights.shape()[3] {
            weights_cells[weight_row_idx].push(weights[[chan_out, ci, cj, ck]].clone());
          }
        }
      }
      weight_row_idx += 1;
    }

    // (O_H * O_W x inp_channels * C_H * C_W)
    for batch in 0..inp.shape()[0] {
      for i in 0..oh {
        for j in 0..ow {
          inp_cells.push(vec![]);
          for ci in 0..weights.shape()[1] {
            for cj in 0..weights.shape()[2] {
              for ck in 0..weights.shape()[3] {
                let idx_i = i * si + ci;
                let idx_j = j * sj + cj;
                inp_cells[input_row_idx].push(inp_pad[[batch, idx_i, idx_j, ck]].clone());
              }
            }
          }
          input_row_idx += 1;
        }
      }
    }

    for _batch in 0..inp.shape()[0] {
      for _ in 0..oh {
        for _ in 0..ow {
          for chan_out in 0..weights.shape()[0] {
            if tensors.len() == 3 {
              biases_cells.push(biases[chan_out].clone());
            } else {
              biases_cells.push((zero.clone(), F::ZERO));
            }
          }
        }
      }
    }

    (inp_cells, weights_cells, biases_cells)
  }

  pub fn splat_depthwise<G: Clone>(
    &self,
    tensors: &Vec<Array<(Rc<G>, F), IxDyn>>,
    zero: Rc<G>,
  ) -> (Vec<Vec<(Rc<G>, F)>>, Vec<Vec<(Rc<G>, F)>>, Vec<(Rc<G>, F)>) {
    let input = &tensors[0];
    let weights = &tensors[1];
    let biases = &tensors[2];

    assert_eq!(tensors.len(), 3);
    assert_eq!(input.shape().len(), 4);
    assert_eq!(weights.shape().len(), 4);
    assert_eq!(input.shape()[0], 1);

    let conv_config = &Self::param_vec_to_config(self.config.layer_params.clone());
    let strides = conv_config.stride;

    let h: usize = input.shape()[1];
    let w: usize = input.shape()[2];
    let ch: usize = weights.shape()[1];
    let cw: usize = weights.shape()[2];
    let (si, sj) = conv_config.stride;
    let (oh, ow) = Self::out_hw(h, w, si, sj, ch, cw, conv_config.padding);

    let (ph, pw) = if conv_config.padding == PaddingEnum::Same {
      Self::get_padding(h, w, si, sj, ch, cw)
    } else {
      ((0, 0), (0, 0))
    };

    let padding = vec![[0, 0], [ph.0, ph.1], [pw.0, pw.1], [0, 0]];

    let inp_pad = pad(&input, padding, &(zero, F::ZERO));

    let mut inp_cells = vec![];
    let mut weight_cells = vec![];
    let mut biases_cells = vec![];
    let mut row_idx = 0;

    for i in 0..oh {
      for j in 0..ow {
        for chan_out in 0..weights.shape()[3] {
          inp_cells.push(vec![]);
          weight_cells.push(vec![]);
          biases_cells.push(biases[[chan_out]].clone());

          for ci in 0..weights.shape()[1] {
            for cj in 0..weights.shape()[2] {
              let idx_i = i * strides.0 + ci;
              let idx_j = j * strides.1 + cj;

              inp_cells[row_idx].push(inp_pad[[0, idx_i, idx_j, chan_out]].clone());
              weight_cells[row_idx].push(weights[[0, ci, cj, chan_out]].clone());
            }
          }

          row_idx += 1;
        }
      }
    }

    (inp_cells, weight_cells, biases_cells)
  }
}

impl<F: PrimeField> Layer<F> for Conv2DChip<F> {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, Rc<AssignedCell<F, F>>>,
    rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let conv_config = &Self::param_vec_to_config(self.config.layer_params.clone());
    let zero = constants.get(&0).unwrap();

    let inp = &tensors[0];
    let weights = &tensors[1];

    let (oh, ow) = Self::out_hw(
      inp.shape()[1],
      inp.shape()[2],
      conv_config.stride.0,
      conv_config.stride.1,
      weights.shape()[1],
      weights.shape()[2],
      conv_config.padding,
    );
    let batch_size = inp.shape()[0];

    let (splat_inp, splat_weights, splat_biases) = match conv_config.conv_type {
      ConvLayerEnum::Conv2D => self.splat(tensors, zero.clone()),
      ConvLayerEnum::DepthwiseConv2D => self.splat_depthwise(tensors, zero.clone()),
    };

    let outp_flat: Vec<(AssignedCell<F, F>, F)> = match conv_config.conv_type {
      ConvLayerEnum::Conv2D => {
        let fc_chip = FullyConnectedChip::<F> {
          _marker: PhantomData,
          config: FullyConnectedConfig::construct(false),
        };

        let conv_size = splat_inp[0].len();
        let flattened_inp: Vec<_> = splat_inp.into_iter().flat_map(|x| x.into_iter()).collect();
        let flattened_weights = splat_weights
          .into_iter()
          .flat_map(|x| x.into_iter())
          .collect::<Vec<_>>();

        let out_channels = weights.shape()[0];
        let inp_array =
          Array::from_shape_vec(IxDyn(&vec![batch_size * oh * ow, conv_size]), flattened_inp)
            .unwrap();
        let weights_array =
          Array::from_shape_vec(IxDyn(&vec![out_channels, conv_size]), flattened_weights).unwrap();

        let outp_slice = fc_chip
          .forward(
            layouter.namespace(|| ""),
            &vec![weights_array, inp_array],
            constants,
            rand_vector,
            gadget_config.clone(),
            layer_config,
          )
          .unwrap();

        let outp_flat = outp_slice[0]
          .t()
          .into_iter()
          .map(|x| (x.0.as_ref().clone(), x.1).clone())
          .collect::<Vec<_>>();
        outp_flat
      }
      ConvLayerEnum::DepthwiseConv2D => {
        // Do the dot products
        let dot_prod_chip = DotProductChip::<F>::construct(gadget_config.clone());
        let mut outp_flat = vec![];
        for (inp_vec, weight_vec) in splat_inp.iter().zip(splat_weights.iter()) {
          let inp_vec = inp_vec.iter().map(|x| (x.0.as_ref(), x.1)).collect::<Vec<_>>();
          let weight_vec = weight_vec.iter().map(|x| (x.0.as_ref(), x.1)).collect::<Vec<_>>();
          let vec_inputs = vec![inp_vec, weight_vec];
          let constants = vec![(zero.as_ref(), F::ZERO)];
          let outp = dot_prod_chip
            .forward(layouter.namespace(|| "dot_prod"), &vec_inputs, &constants)
            .unwrap();
          outp_flat.push(outp[0].clone());
        }
        // println!("outp_flat: {:?}", outp_flat.len());

        outp_flat
      }
    };

    let mut biases = vec![];
    for bias in splat_biases.iter() {
      biases.push((bias.0.as_ref(), bias.1));
    }

    // Compute the bias + div + relu
    let bdr_chip = BiasDivRoundRelu6Chip::<F>::construct(gadget_config.clone());
    let tmp = vec![(zero.as_ref(), F::ZERO)];
    let outp_flat = outp_flat.iter().map(|x| (&x.0, x.1)).collect::<Vec<_>>();
    let outp = bdr_chip
      .forward(
        layouter.namespace(|| "bias_div_relu"),
        &vec![outp_flat, biases],
        &tmp,
      )
      .unwrap();

    // TODO: this is also horrible. The bdr chip outputs interleaved [(relu'd, div'd), (relu'd, div'd), ...]
    // Uninterleave depending on whether or not we're doing the relu
    let outp = if conv_config.activation == ActivationType::Relu6 {
      outp
        .into_iter()
        .step_by(2)
        .map(|x| (Rc::new(x.0), x.1))
        .collect::<Vec<_>>()
    } else if conv_config.activation == ActivationType::None {
      outp
        .into_iter()
        .skip(1)
        .step_by(2)
        .map(|x| (Rc::new(x.0), x.1))
        .collect::<Vec<_>>()
    } else if conv_config.activation == ActivationType::Relu {
      let dived = outp.iter().skip(1).step_by(2).map(|x| (&x.0, x.1)).collect::<Vec<_>>();
      let relu_chip = ReluChip::<F>::construct(gadget_config.clone());
      let relu_outp = relu_chip
        .forward(layouter.namespace(|| "relu"), &vec![dived], &tmp)
        .unwrap();
      let relu_outp = relu_outp
        .into_iter()
        .map(|x| (Rc::new(x.0), x.1))
        .collect::<Vec<_>>();
      relu_outp
    } else {
      panic!("Unsupported activation type");
    };

    let oc = match conv_config.conv_type {
      ConvLayerEnum::Conv2D => weights.shape()[0],
      ConvLayerEnum::DepthwiseConv2D => weights.shape()[3],
    };

    let out_shape = vec![batch_size, oh, ow, oc];
    let outp = Array::from_shape_vec(IxDyn(&out_shape), outp).unwrap();

    Ok(vec![outp])
  }
}

impl<F: PrimeField> GadgetConsumer for Conv2DChip<F> {
  fn used_gadgets(&self, layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    let conv_config = &Self::param_vec_to_config(layer_params.clone());
    let mut outp = vec![
      GadgetType::Adder,
      GadgetType::DotProduct,
      GadgetType::InputLookup,
      GadgetType::BiasDivRoundRelu6,
    ];

    if conv_config.activation == ActivationType::Relu {
      outp.push(GadgetType::Relu);
    }

    outp
  }
}
