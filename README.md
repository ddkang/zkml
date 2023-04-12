# zkml

zkml is a framework for constructing proofs of ML model execution in ZK-SNARKs.
Read our [blog
post](https://medium.com/@danieldkang/trustless-verification-of-machine-learning-6f648fd8ba88)
and [paper](https://arxiv.org/abs/2210.08674) for implementation details.

zkml requires the nightly build of Rust:

```
rustup override set nightly
```

## Quickstart

Run the following commands:

```sh
# Installs rust, skip if you already have rust installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

git clone https://github.com/ddkang/zkml.git
cd zkml
rustup override set nightly
cargo build --release
mkdir params_kzg
mkdir params_ipa

# This should take ~16s to run the first time
# and ~8s to run the second time
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg
```

This will prove an MNIST circuit! It will require around 2GB of memory and take
around 8 seconds to run.

## Contact us

If you're interested in extending or using zkml, please contact us at `ddkang
[at] g.illinois.edu`.
