
import os, sys, subprocess, shutil
scriptDir = os.path.dirname(os.path.realpath(__file__))

from . import config, parallel
from .scalar import *
_dtype = dtype

def graphGetChildren(outputs):
    children = {}

    def _childrenFunc(out):
        for inp in out.args:
            if inp in children:
                children[inp] += 1
            else:
                children[inp] = 1
                _childrenFunc(inp)

    output = Container()
    output.args = tuple(outputs)
    _childrenFunc(output)
    for out in outputs:
        children[out] -= 1
    return children

def graphTopologicalSort(outputs, children):
    output = Container()
    output.args = tuple(outputs)
    for out in outputs:
        children[out] += 1
    children[output] = 0
    sortedOps = []
    def _sort(out):
        sortedOps.append(out)
        for inp in out.args:
            children[inp] -= 1
            if children[inp] == 0:
                _sort(inp)
    _sort(output)
    return sortedOps[1:][::-1]


class Variable(ArithBase):
#class Variable(object):
    _index = 0
    def __init__(self, shape, dtype=_dtype):
        index = Variable._index
        Variable._index += 1
        self.name = 'Variable_{}'.format(index)
        self.shape = shape
        self.args = ()
        self.index = 0
        self.dtype = dtype
        self.outputIndex = None

    def __getitem__(self, index):
        #print self.index
        #assert self.index == 0
        var = self.getReference()
        var.index = index
        return var

    def getReference(self):
        var = Variable(self.shape, self.dtype)
        var.args = (self,)
        var.name = self.name
        return var

    def grad(self, grad):
        assert isinstance(grad, tuple)
        for out in grad:
            assert self.shape == out.shape
            assert self.dtype == out.dtype
        index = 0
        if len(self.args) == 0:
            return tuple()
        elif isinstance(self.args[0], Variable):
            index = self.args[0].index
        gradArgs = (grad[0][index],)
        gradArgs[0].args = grad
        if isinstance(self.args[0], TensorFunctionOp):
            gradArgs[0].outputIndex = self.outputIndex
        return gradArgs

class Zeros(Variable):
    pass

import inspect
class TensorFunctionOp(object):
    _gradCache = {}
    def __init__(self, func, args, outputs, indices):
        assert isinstance(indices, IntegerScalar) or isinstance(indices, int)
        self.func = func
        self.name = func.name
        n = len(self.func._inputTensors)
        self.indices = indices
        self.info = inspect.stack()[2:]
        args = args + outputs
        self.args = args
        self.outputs = []
        for index, out in enumerate(outputs):
            self.outputs.append(out.getReference())
            self.outputs[-1].args = (self,)
            self.outputs[-1].outputIndex = index
        self.outputs = tuple(self.outputs)

    def getCallString(self):
        #callString = '\n/* ' + str(self.info) + ' */\n'
        callString = ''
        for inp in self.args:
            if isinstance(inp.index, int):
                offset = '({})'.format(inp.index)
            else:
                offset = '({})'.format(inp.index.name)
            callString += '&{}{}, '.format(inp.name, offset)
            #if hasattr(inp, 'dtype'):
            #    callString += '/* {} */  '.format(inp.dtype)
        #for out in self.outputs:
        #    callString += '{}, '.format(out.name)
        return callString[:-2]

    def grad(self, grad):
        inputs = self.args[:-len(self.outputs)]
        args = inputs
        outputs = list(grad)
        for index in (set(range(0, len(self.outputs)))-set([x.outputIndex for x in grad])):
            out = self.outputs[index]
            outputs.append(Zeros(out.shape, out.dtype))
            outputs[-1].outputIndex = index
        outputs = list(sorted(outputs, key=lambda x: x.outputIndex))
        assert len(outputs) == len(self.outputs)
        for out1, out2 in zip(outputs, self.outputs):
            assert out1.shape == out2.shape
            assert out1.dtype == out2.dtype
        cache = TensorFunctionOp._gradCache
        for inp in inputs:
            if inp.name in cache:
                out = cache[inp.name]
            else:
                out = Zeros(inp.shape, inp.dtype)
                cache[inp.name] = out
            outputs.append(out[inp.index])
        outputs = tuple(outputs)
        #print self.name, len(self.outputs), [(x.name, x.dtype) for x in self.outputs], [(x.name, x.dtype) for x in grad], [(x.name, x.dtype) for x in outputs[:len(self.outputs)]]
        return TensorFunctionOp(self.func.grad, args, outputs, self.indices).outputs

    @classmethod
    def clear_cache(self):
        TensorFunctionOp._gradCache = {}

class ExternalFunctionOp(object):
    def __init__(self, name, args, outputs, empty=False):
        self.name = 'Function_' + name
        print self.name
        args = args + outputs
        self.args = args
        self.empty = empty
        self.outputs = tuple([x.getReference() for x in outputs])
        for out in self.outputs:
            out.args = (self,)

    def getCallString(self):
        if self.empty:
            return ''
        inp = self.args[0]
        shape = ','.join([str(x) for x in inp.shape[1:]])
        callString = 'std::vector<arrType<{}, {}>*>{{'.format(inp.dtype, shape)
        for inp in self.args:
            callString += '&{},'.format(inp.name)
        callString = callString[:-1] + '}'
        return callString

    def grad(self, grad):
        name = self.name[len('Function_'):] + '_grad'
        inputs = self.args[:-len(self.outputs)]
        args = inputs
        outputs = list(grad)
        assert len(outputs) == len(self.outputs)
        cache = TensorFunctionOp._gradCache
        for inp in inputs:
            if inp.name in cache:
                out = cache[inp.name]
            else:
                out = Zeros(inp.shape, inp.dtype)
                cache[inp.name] = out
            outputs.append(out)
        outputs = tuple(outputs)
        return ExternalFunctionOp(name, args, outputs, self.empty).outputs

class Function(object):
    _index = 0
    _module = None
    codeDir = os.path.dirname(__file__) + '/gencode/'
    funcs = []

    def __init__(self, name, inputs, outputs, grad=True):
        self.name = name
        self._inputs = inputs
        self._outputs = outputs
        _outputs = [x for x in self._outputs if x is not None]
        self._children = graphGetChildren(_outputs)
        if config.compile:
            self._genCode(_outputs)
            TensorFunctionOp.clear_cache()
            if grad:
                self.grad = self._getAdjoint()

        Function.funcs.append(self.name)

    def _getAdjoint(self):
        #gradOutputs = []
        #for out in self._outputTensors:
        #    gradOutputs.append(Tensor(out.shape))
        #    gradOutputs[-1].cellTensor = out.cellTensor

        #scalarOutput = sum([x.dot(y) for x, y in zip(self._outputTensors, gradInputs)])
        gradients = {}
        gradOutputs = []
        cache = TensorFunctionOp._gradCache
        for out in self._outputs:
            grad = Variable(out.shape, out.dtype)
            cache[out.name] = grad
            gradients[out] = (grad,)
            gradOutputs.append(grad)
        gradInputs = self._diff(self._outputs, self._inputs, gradients)
        return Function(self.name + '_grad', self._inputs + gradOutputs, gradInputs, grad=False)
        
    def _genCode(self, outputs):
        codeFile = open(self.codeDir + 'code.cpp', 'a')
        codeFile.write('\nstatic PyObject* Function_{}(PyObject *self, PyObject *args) {{\n'.format(self.name))
        #for out in self._outputs:
        #    memString += '{}* {}, '.format(out.dtype, out.name)
        for index, inp in enumerate(self._inputs):
            if isinstance(inp, IntegerScalar):
                codeFile.write('\tinteger {} = (integer) PyInt_AsLong(PyTuple_GetItem(args, {}));\n'.format(inp.name, index))
                continue
            codeFile.write('\tPyObject* Py_{} = PyTuple_GetItem(args, {});\n'.format(inp.name, index))
            shape = ','.join([str(x) for x in inp.shape[1:]])
            codeFile.write('\tarrType<{}, {}> {};\n'.format(inp.dtype, shape, inp.name))
            codeFile.write('\tgetArray((PyArrayObject*) Py_{0}, {0});\n'.format(inp.name))
        codeFile.write('\n')

        #codeFile.write('\nvoid Function_{}({}) {}\n'.format(self.name, memString[:-2], '{\n'))

        sortedOps = graphTopologicalSort(outputs, self._children.copy())
        def _getName(op):
            if isinstance(op, int):
                name = op
            else:
                name = op.name
            return name
        for op in sortedOps:
            if isinstance(op, Zeros):
                assert len(op.args) == 0
                #codeFile.write('\t// init var {}\n'.format(op.name)) 
                shape = ','.join([str(x) for x in op.shape[1:]])
                codeFile.write('\tarrType<{}, {}> {}({}, true);\n'.format(op.dtype, shape, op.name, _getName(op.shape[0]))) 
            elif isinstance(op, TensorFunctionOp):
                codeFile.write('\t{}({}, {});\n'.format(op.name, _getName(op.indices), op.getCallString()))
            elif isinstance(op, ExternalFunctionOp):
                codeFile.write('\t{}({});\n'.format(op.name, op.getCallString()))
            elif not isinstance(op, Variable):
                raise Exception('op not recognised', op)

        codeFile.write('\n\tPyObject* outputs = PyTuple_New({});\n'.format(len(outputs)))
        for index, out in enumerate(outputs):
            codeFile.write('\tPyTuple_SetItem(outputs, {}, putArray({}));\n'.format(index, out.name))
        codeFile.write('\treturn outputs;')
        codeFile.write('\n')
        codeFile.write('}\n\n')

        codeFile.close()

    def _diff(self, outputs, inputs, gradients=None):
        if gradients is None:
            gradients = {}
            for out in outputs:
                gradients[out] = 1.
        children = self._children.copy()
        #print children.values()
        def _diffFunc(out):
            assert children[out] == 0
            grads = []
            if gradients[out] == None:
                grads = tuple([None]*len(out.args))
            else:
                grads = out.grad(gradients[out])
            #elif hasattr(out, 'grad'):
            assert len(grads) == len(out.args)
            for grad, inp in zip(grads, out.args):
                if isinstance(inp, TensorFunctionOp):
                    assert grad.dtype == 'scalar'
                if grad is None and inp not in gradients:
                    gradients[inp] = None
                elif inp not in gradients:
                    gradients[inp] = (grad,)
                else:
                    gradients[inp] += (grad,)
                children[inp] -= 1
                if children[inp] == 0:
                    _diffFunc(inp)
        for out in outputs:
            if children[out] == 0:
                _diffFunc(out)
        #print children.values()
        return [gradients.get(inp, (None,))[0] for inp in inputs]

    def __call__(self, *args):
        func = getattr(Function._module, self.name)
        return func(*args)

    @classmethod
    def createCodeDir(self, case):
        self.codeDir = case + 'gencode/'
        if config.compile:
            assert not os.path.exists(self.codeDir)
            shutil.copytree(scriptDir + '/gencode', self.codeDir)

    @classmethod
    def clean(self):
        if not config.compile:
            return
        #try:
        #    os.remove(self.codeDir + 'code.cpp')
        #except:
        #    pass
        with open(self.codeDir + 'code.cpp', 'a') as f:
            f.write('#include "code.hpp"\n')

    @classmethod
    def compile(self):
        if config.compile:
            with open(self.codeDir + 'code.cpp', 'a') as f:
                f.write("""PyMethodDef Methods[] = {
        {"initialize",  initSolver, METH_VARARGS, "Execute a shell command."},
        {"viscosity",  viscosity, METH_VARARGS, "Execute a shell command."},
        {"finalize",  finalSolver, METH_VARARGS, "Execute a shell command."},
""")

                for name in Function.funcs:
                    f.write('\t{{"{0}", Function_{0}, METH_VARARGS, "boo"}},\n'.format(name))
                #f.write('\n\n' + self.extraCode)
                f.write("""
                {NULL, NULL, 0, NULL}        /* Sentinel */
        };
""")
            subprocess.check_call(['make'], cwd=self.codeDir)
        parallel.mpi.Barrier()
        sys.path.append(self.codeDir)
        import interface as mod
        Function._module = mod

