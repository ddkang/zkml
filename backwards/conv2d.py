
class Conv2D():
    def __init__(self, input_shape, kernel_size, strides, padding, activation, name):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.name = name

        self.input_idxes = []
        self.output_idxes = []

    # This is called in order to add new layers to the transcript
    def backward(dag):
        # Add the input layer
        transcript.add_layer(Conv2D(self.input_shape, self.kernel_size, self.strides, self.padding, self.activation, self.name))