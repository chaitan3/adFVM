class Function:
    def __init__(self, inputs, outputs):
        self._inputs = self._discover_inputs(outputs)
        self._inputs_map = {}
        for index, inp in enumerate(inputs):
            try:
                self._inputs_map[index] = self._inputs.index(inp)
            except ValueError:
                print('Excess input specified')
        self._outputs = outputs
        return 

    def __call__(self, *inputs):
        self._clear_values()

        for index, inp in enumerate(inputs):
            try:
                self._inputs[self._inputs_map[index]].value = inputs[index]
            except KeyError:
                pass

        return [output.value for output in self._outputs]

    def _clear_values(self):
        return

    def _discover_inputs(self, outputs, inputs=[]):
        for out in outputs:
            op = out.parent
            if op is not None:
                self._discover_inputs(op.inputs, inputs)
            else:
                inputs.append(out)
        return inputs
