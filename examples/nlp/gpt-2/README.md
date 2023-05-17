# Proving GPT-2 Execution

## Verification

We show how to prove GPT-2 execution with zkml. The GPT-2 model we use is a distilled version of
GPT-2 available [here](https://huggingface.co/distilgpt2/blob/main/64.tflite). Please read the
license for terms of use.

Because proving is resource-intensive we've published a proof in this directory. To verify the
proof, first un-tar the `public_vals` and `vkey` in _this_ directory:
```
tar -xvf public_vals.tar.gz
tar -xvf vkey.tar.gz
```

Then download the parameters from [here](https://drive.google.com/file/d/1bhAYXOzMnAI-tB6VbUkCY7tThQ9L5K6G/view?usp=sharing).

After downloading the files, execute the following commands in the root directory:
```
# Before you execute the proof, make sure you've followed the installation instructions

./target/release/verify_circuit examples/nlp/gpt-2/config.msgpack examples/nlp/gpt-2/vkey examples/nlp/gpt-2/proof examples/nlp/gpt-2/public_vals kzg
```


## Proving

In order to prove GPT-2 execution, first download the model weights from
[here](https://drive.google.com/file/d/1XKeuJvXp_c-Xm4seN4ZNJobc4w-GRLlA/view?usp=sharing). Make
sure the weights are downloaded to _this_ directory. Make sure to download the parameters as
described from above as well. Since the model weights were derived from the Huggingface repository,
please read the license for terms of use.

Then, in the _root_ directory execute the following commands:
```
./target/release/time_circuit examples/nlp/gpt-2/model.msgpack examples/nlp/gpt-2/inp.msgpack kzg
```
This will produce `public_vals`, `pkey`, `vkey`, and `proof`. You should be able to use these to
verify the proof.
