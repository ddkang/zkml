# Generate a mock input for the model

import numpy as np
import os

# Generate a mock numpy array with the specified size and type
input_shape = (1, 224, 224, 3)
mock_input = np.random.rand(*input_shape).astype(np.float32)

# Save the mock input to a file
output_dir = "generated/mobilenet_v2_1.0_224"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.save(f"{output_dir}/input.npy", mock_input)
print(f"Mock input saved at {output_dir}/mock_input.npy")

