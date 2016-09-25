class Op(object):
    def __init__(self, inputs=[]):
        self.outputs = self.make_node(inputs)
        # make_node can modify inputs
        self.inputs = inputs
        for output in self.outputs:
            output.parent = self
        return

    def _get_input_values(self):
        return [inp.value for inp in self.inputs]

    def _set_output_values(self, output_values):
        for index, value in enumerate(output_values):
            self.outputs[index].value = value
        return

    def py_compute(self):
        #print 'computing', self
        input_values = self._get_input_values()
        output_values = self.perform(input_values)
        self._set_output_values(output_values)
        return

    def make_node(self, inputs):
        raise Exception('Not implemented')

    def perform(self, inputs):
        raise Exception('Not implemented')

    def c_code(self):
        raise Exception('Not implemented')

