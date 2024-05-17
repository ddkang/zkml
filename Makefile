
experiment1_generate:
	python3 tf_models/mobilenet-v2-1.0-224/generate-tflite.py && python3 tf_models/mobilenet-v2-1.0-224/generate-input.py

experiment1_convert_model:
	python3 python/converter.py --model generated/mobilenet_v2_1.0_224/model.tflite --model_output generated/mobilenet_v2_1.0_224/converted_model.msgpack --config_output generated/mobilenet_v2_1.0_224/config.msgpack --scale_factor 512 --k 24 --num_cols 10 --num_randoms 12544

experiment1_convert_inputs:
	python3 python/input_converter.py --model_config generated/mobilenet_v2_1.0_224/config.msgpack --inputs generated/mobilenet_v2_1.0_224/input.npy --output generated/mobilenet_v2_1.0_224/input.msgpack

experiment1_test:
	./target/release/test_circuit generated/mobilenet_v2_1.0_224/converted_model.msgpack generated/mobilenet_v2_1.0_224/input.msgpack





experiment_setup:
	mkdir -p generated/$(MODEL_NAME) && make $(EXPERIMENT_NAME)_generate && make $(EXPERIMENT_NAME)_convert_model && make $(EXPERIMENT_NAME)_convert_inputs
