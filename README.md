# zkml

`zkml` requires the nightly build of Rust:
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

# This should take ~16s to run the first time and ~8s to run the second time
./target/release/time_circuit examples/mnist/model.msgpack examples/mnist/inp.msgpack kzg
```

This will prove an MNIST circuit! It will require around 5GB of memory and take around 16 seconds to
run.
