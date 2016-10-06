import numpy as np
from numbers import Number

import sys
import subprocess

class Function:
    def __init__(self, inputs, outputs, mode='py'):
        # TODO
        # broadcasting
        # shapeOp
        # integer arrays
        # in place optimizations
        # garbage collection
        self._sorted_ops = self._topological_sort(outputs)
        self._inputs = self._discover_inputs(outputs)
        self._inputs_map = self._get_inputs_map(inputs)
        self._rev_inputs_map = {v: k for k, v in self._inputs_map.items()}
        self._outputs = outputs
        if mode == 'c':
            self._gen_c_code()
            self._import_c_function()
        else:
            assert mode =='py'
        self._mode = mode
        return 

    def __call__(self, *inputs, **kwargs):
        if self._mode == 'py':
            return self._py_call(*inputs, **kwargs)
        else:
            return self._c_call(*inputs)

    def _py_call(self, *inputs, **kwargs):
        self._clear_values()
        for index, inp in enumerate(inputs):
            try:
                _input = self._inputs[self._inputs_map[index]]
                value = inputs[index]
                self._check_input(_input, value)
                _input.value = value
            except KeyError:
                pass
        outputs = [output.value for output in self._outputs]
        return outputs

    def _gen_c_code(self):
        n_inputs = len(self._inputs)
        n_outputs = len(self._outputs)
        c_code = '''
#define FUNCTION_INTERFACE 1

PyObject *interface(PyObject* self, PyObject *args) {{

    int n_inputs = PyTuple_Size(args);
    int n_outputs = %(n_outputs)s;
    int64_t i0, i1, i2, i3;

    {init_code}    

    {graph_code}


    PyObject* outputs = PyList_New(n_outputs);

    {final_code}
    
    return outputs;
}}
''' % {'n_outputs': n_outputs}

        init_code = ''
        for index, inp in enumerate(self._inputs):
            mapped_index = self._rev_inputs_map[index]
            mapped_input = self._inputs[index]
            init_code += '''
    ndarray *ndarr_{inp} = ndarray_from_numpy(PyTuple_GetItem(args, {index}));
        '''.format(inp=mapped_input.id, index=mapped_index)

        graph_code = ''
        for op in self._sorted_ops:
            graph_code += op.c_generate()

        final_code = ''
        for index, out in enumerate(self._outputs):
            final_code += '''
    PyList_SetItem(outputs, {index}, numpy_from_ndarray(ndarr_{out}));
        '''.format(out=out.id, index=index)

        c_code = c_code.format(
            init_code = init_code,
            graph_code = graph_code,
            final_code = final_code
            )
        with open('interface.c', 'w') as f:
            f.write(c_code)
        return

    def _import_c_function(self):
        subprocess.check_call([sys.executable, 'setup.py', 'build_ext', '--inplace'])
        import function
        self._c_call = function.interface
        return

    def _clear_values(self):
        return

    def _check_input(self, _input, value):
        if isinstance(value, np.ndarray):
            assert _input.dims == len(value.shape)
            assert value.flags['C_CONTIGUOUS']
        elif isinstance(value, Number):
            assert _input.dims == 0
        else:
            raise Exception('input not recognized')
        return


    def _topological_sort(self, outputs, sorted_ops=[]):
        for out in outputs:
            op = out.parent
            if op is not None:
                self._topological_sort(op.inputs, sorted_ops)
                sorted_ops.append(op)
        return sorted_ops

    def _discover_inputs(self, outputs, inputs=[]):
        for out in outputs:
            op = out.parent
            if op is not None:
                self._discover_inputs(op.inputs, inputs)
            else:
                inputs.append(out)
        return inputs

    def _get_inputs_map(self, inputs):
        inputs_map = {}
        for index, inp in enumerate(inputs):
            try:
                inputs_map[index] = self._inputs.index(inp)
            except ValueError:
                print 'Excess input specified at', index
        return inputs_map

