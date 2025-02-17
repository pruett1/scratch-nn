class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # compute output Y given input X
    def forward_prop(self, input):
        raise NotImplementedError
    
    # compute dE/dX given dE/dY and update params
    def backward_prop(self, output_error, learning_rate):
        raise NotImplementedError