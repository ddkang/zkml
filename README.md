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



## Converting your own model and data

To convert your own model and data, you will need to convert the model and data to the format zkml
expects. Currently, we accept TFLite models. We show an example below.

1. First `cd examples/mnist`

2. We've already created a model that achieves high accuracy on MNIST (`model.tflite`). You will
   need to create your own TFLite model. One way is to [convert a model from Keras](https://stackoverflow.com/questions/53256877/how-to-convert-kerash5-file-to-a-tflite-file).

3. You will need to convert the model:
```bash
python ../../python/converter.py --model model.tflite --model_output converted_model.msgpack --config_output config.msgpack --scale_factor 512 --k 17 --num_cols 10 --num_randoms 1024
```

There are several parameters that need to be changed depending on the model (`scale_factor`, `k`,
`num_cols`, and `num_randoms`).

4. You will first need to serialize the model input to numpy's serialization format `npy`. We've
   written a small script to do this for the first test data point in MNIST:
```bash
python data_to_npy.py
```

5. You will then need to convert the input to the model:
```bash
python ../../python/input_converter.py --model_config converted_model.msgpack --inputs 7.npy --output example_inp.msgpack
```

6. Once you've converted the model and input, you can run the model as above! However, we generally
   recommend testing the model before proving (you will need to build zkml before running the next
   line):
```bash
cd ../../
./target/release/test_circuit examples/mnist/converted_model.msgpack examples/mnist/example_inp.msgpack
```


## Contact us

If you're interested in extending or using zkml, please contact us at `ddkang
[at] g.illinois.edu`.
