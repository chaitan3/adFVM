class Op(object):
    def __init__(self, inputs=[]):
        self.outputs = self.make_node(inputs)
        # make_node can modify inputs
        self.inputs = inputs
        for output in self.outputs:
            output.parent = self
        return

    def py_compute(self):
        #print 'computing', self
        input_values = [inp.value for inp in self.inputs]
        output_values = self.perform(input_values)
        for index, value in enumerate(output_values):
            self.outputs[index].value = value
        return

    def c_generate(self):
        repl = {}
        for i in range(0, len(self.inputs)):
            repl['input_{}'.format(i)] = 'ndarr_{}'.format(self.inputs[i].id)
        for i in range(0, len(self.outputs)):
            repl['output_{}'.format(i)] = 'ndarr_{}'.format(self.outputs[i].id)
        return self.c_code().format(**repl)

    def make_node(self, inputs):
        raise Exception('Not implemented', self)

    def perform(self, inputs):
        raise Exception('Not implemented', self)

    def c_code(self):
        raise Exception('Not implemented', self)

