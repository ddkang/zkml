# Proving CLIP execution

## Verification

We show how to prove CLIP execution with zkml. The CLIP model we use was converted by Keras for
Stable Diffusion (link
[here](https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/)). Please read the
CLIP license for terms of use.

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

./target/release/verify_circuit examples/nlp/clip/config.msgpack examples/nlp/clip/vkey examples/nlp/clip/proof examples/nlp/clip/public_vals kzg
```


## Proving

In order to prove CLIP execution, first download the model weights from
[here](https://drive.google.com/file/d/1GkvF8x4sG9DpGQv88xOitVRRg34q3fdN/view?usp=sharing). Make
sure the weights are downloaded to _this_ directory. Make sure to download the parameters as
described  from above as well. Since the model weights were derived from the Keras version, please
read the license for terms of use.

Then, in the _root_ directory execute the following commands:
```
./target/release/time_circuit examples/nlp/clip/model.msgpack examples/nlp/clip/inp.msgpack kzg
```
This will produce `public_vals`, `pkey`, `vkey`, and `proof`. You should be able to use these to
verify the proof.
