import numpy as np
from numbers import Number

import setup

class Function:
    def __init__(self, inputs, outputs):
        self._inputs = self._discover_inputs(outputs)
        self._inputs_map = {}
        for index, inp in enumerate(inputs):
            try:
                self._inputs_map[index] = self._inputs.index(inp)
            except ValueError:
                print 'Excess input specified at', index
        self._outputs = outputs
        return 

    def __call__(self, *inputs, **kwargs):
        self._clear_values()
        for index, inp in enumerate(inputs):
            try:
                _input = self._inputs[self._inputs_map[index]]
                value = inputs[index]
                self._check_input(_input, value)
                _input.value = value
            except KeyError:
                pass
        return [output.value for output in self._outputs]

    def _check_input(self, _input, value):
        if isinstance(value, np.ndarray):
            assert _input.dims == len(value.shape)
            assert value.flags['C_CONTIGUOUS']
        elif isinstance(value, Number):
            assert _input.dims == 0
        else:
            raise Exception('input not recognized')
        return

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
