use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    bias_div_round_relu6::BiasDivRoundRelu6Chip,
    dot_prod::DotProductChip,
    gadget::{Gadget, GadgetConfig},
  },
  layers::pad::pad,
};

use super::layer::{Layer, LayerConfig};

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
  pub do_relu: bool,
  pub stride: (usize, usize),
}

pub struct Conv2DChip<F: FieldExt> {
  pub config: LayerConfig,
  pub _marker: PhantomData<F>,
}

impl<F: FieldExt> Conv2DChip<F> {
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
    let do_relu = layer_params[2] == 1;
    let stride = (layer_params[3] as usize, layer_params[4] as usize);
    Conv2DConfig {
      conv_type,
      padding,
      do_relu,
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
      (ci - sj).max(0)
    } else {
      (ci - (h % si)).max(0)
    };
    let pw = if w % sj == 0 {
      (cj - sj).max(0)
    } else {
      (cj - (w % sj)).max(0)
    };
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
    println!(
      "H: {}, W: {}, SI: {}, SJ: {}, CH: {}, CW: {}",
      h, w, si, sj, ch, cw
    );
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
    tensors: &Vec<Array<G, IxDyn>>,
    zero: G,
  ) -> (Vec<Vec<G>>, Vec<Vec<G>>, Vec<G>) {
    assert_eq!(tensors.len(), 3);

    let conv_config = &Self::param_vec_to_config(self.config.layer_params.clone());

    let inp = &tensors[0];
    let weights = &tensors[1];
    let biases = &tensors[2];

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
    println!("Padding: {:?}", (ph, pw));
    let padding = vec![[0, 0], [ph.0, ph.1], [pw.0, pw.1], [0, 0]];

    let inp_pad = pad(&inp, padding, zero);

    // We only support valid padding with stride = 1 at the moment
    /*
    if conv_config.padding == PaddingEnum::Valid {
      assert_eq!(si, 1);
      assert_eq!(sj, 1);
    }
    */

    let (oh, ow) = Self::out_hw(h, w, si, sj, ch, cw, conv_config.padding);

    let mut inp_cells = vec![];
    let mut weights_cells = vec![];
    let mut biases_cells = vec![];
    let mut row_idx = 0;
    for i in 0..oh {
      for j in 0..ow {
        for chan_out in 0..weights.shape()[0] {
          inp_cells.push(vec![]);
          weights_cells.push(vec![]);
          biases_cells.push(biases[[chan_out]].clone());

          for ci in 0..weights.shape()[1] {
            for cj in 0..weights.shape()[2] {
              for ck in 0..weights.shape()[3] {
                let idx_i = i * si + ci;
                let idx_j = j * sj + cj;

                inp_cells[row_idx].push(inp_pad[[0, idx_i, idx_j, ck]].clone());
                weights_cells[row_idx].push(weights[[chan_out, ci, cj, ck]].clone());
              }
            }
          }

          row_idx += 1;
        }
      }
    }

    (inp_cells, weights_cells, biases_cells)
  }

  pub fn splat_depthwise<G: Clone>(
    &self,
    tensors: &Vec<Array<G, IxDyn>>,
    zero: G,
  ) -> (Vec<Vec<G>>, Vec<Vec<G>>, Vec<G>) {
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

    let inp_pad = pad(&input, padding, zero);

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

impl<F: FieldExt> Layer<F> for Conv2DChip<F> {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    let conv_config = &Self::param_vec_to_config(self.config.layer_params.clone());
    let zero = constants.get(&0).unwrap().clone();

    println!("tensors: {:?}", tensors.len());
    for tensor in tensors {
      println!("tensor: {:?}", tensor.shape());
    }

    let (splat_inp, splat_weights, splat_biases) = match conv_config.conv_type {
      ConvLayerEnum::Conv2D => self.splat(tensors, zero.clone()),
      ConvLayerEnum::DepthwiseConv2D => self.splat_depthwise(tensors, zero.clone()),
    };
    println!("splat_inp: {:?}", splat_inp.len());
    println!("splat_weights: {:?}", splat_weights.len());

    // Do the dot products
    let dot_prod_chip = DotProductChip::<F>::construct(gadget_config.clone());
    let mut outp_flat = vec![];
    let mut biases = vec![];
    for ((inp_vec, weight_vec), bias) in splat_inp
      .iter()
      .zip(splat_weights.iter())
      .zip(splat_biases.iter())
    {
      let inp_vec = inp_vec.iter().map(|x| x).collect::<Vec<_>>();
      let weight_vec = weight_vec.iter().map(|x| x).collect::<Vec<_>>();
      let vec_inputs = vec![inp_vec, weight_vec];
      let constants = vec![zero.clone()];
      let outp =
        dot_prod_chip.forward(layouter.namespace(|| "dot_prod"), &vec_inputs, &constants)?;
      outp_flat.push(outp[0].clone());
      biases.push(bias);
    }
    println!("outp_flat: {:?}", outp_flat.len());

    // Compute the bias + div + relu
    let bdr_chip = BiasDivRoundRelu6Chip::<F>::construct(gadget_config.clone());
    let tmp = vec![zero.clone()];
    let outp_flat = outp_flat.iter().map(|x| x).collect::<Vec<_>>();
    let outp = bdr_chip.forward(
      layouter.namespace(|| "bias_div_relu"),
      &vec![outp_flat, biases],
      &tmp,
    )?;

    // TODO: this is also horrible. The bdr chip outputs interleaved [(relu'd, div'd), (relu'd, div'd), ...]
    // Uninterleave depending on whether or not we're doing the relu
    let outp = if conv_config.do_relu {
      outp.iter().step_by(2).cloned().collect::<Vec<_>>()
    } else {
      outp.iter().skip(1).step_by(2).cloned().collect::<Vec<_>>()
    };

    let inp = &tensors[0];
    println!("inp: {:?}", inp.shape());
    println!("outp: {:?}", outp.len());
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
    let oc = match conv_config.conv_type {
      ConvLayerEnum::Conv2D => weights.shape()[0],
      ConvLayerEnum::DepthwiseConv2D => weights.shape()[3],
    };
    let out_shape = vec![1, oh, ow, oc];
    let outp = Array::from_shape_vec(IxDyn(&out_shape), outp).unwrap();

    Ok(vec![outp])
  }
}
