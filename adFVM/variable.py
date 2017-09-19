
import os, sys, subprocess, shutil
scriptDir = os.path.dirname(os.path.realpath(__file__))

from . import config
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
    _index = 0
    def __init__(self, shape, dtype=_dtype):
        index = Variable._index
        Variable._index += 1
        self.name = 'Variable_{}'.format(index)
        self.shape = shape
        self.dtype = dtype
        self.args = ()
        self.index = 0
        self.outputIndex = None
        self.static = False

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
        var.static = self.static
        return var

    def grad(self, grad):
        assert isinstance(grad, tuple)
        assert len(grad) == 1
        gradIndex = 0
        for out in grad:
            assert self.shape == out.shape
            assert self.dtype == out.dtype
        if len(self.args) == 0:
            return tuple()
        assert len(self.args) == 1
        index = 0
        if isinstance(self.args[0], Variable):
            index = self.args[0].index
        gradArgs = (grad[gradIndex][index],)
        if isinstance(self.args[0], FunctionOp):
            gradArgs[0].outputIndex = self.outputIndex
        return gradArgs

class Zeros(Variable):
    pass

class IntegerVariable(Variable):
    def __init__(self, shape, dtype='integer'):
        super(IntegerVariable, self).__init__(shape, dtype)

import inspect
class FunctionOp(object):
    def _init(self, name, args, outputs):
        self.name = name
        args = args + outputs
        self.args = args
        self.outputs = []
        for index, out in enumerate(outputs):
            self.outputs.append(out.getReference())
            self.outputs[-1].args = (self,)
            self.outputs[-1].outputIndex = index
        self.outputs = tuple(self.outputs)

    def _grad(self, grad):
        outputRefs = self.outputs
        n = len(outputRefs)
        _inputs, _outputs = self.args[:-n], self.args[-n:]
        inputs = []
        indices = set([x.outputIndex for x in grad])
        assert all([x.outputIndex is not None for x in grad])
        for out1, out2 in zip(grad, [_outputs[i] for i in indices]):
            inputs.append(out1[out2.index])
            inputs[-1].outputIndex = out1.outputIndex
        extraIndices = set(range(0, n))-indices
        for index in extraIndices:
            out = _outputs[index]
            zero = Zeros(out.shape, out.dtype)
            zero.name = out.name + '_adj'
            inputs.append(zero[out.index])
            inputs[-1].outputIndex = index
            FunctionOp.insert_cache([inputs[-1]])
        inputs = list(sorted(inputs, key=lambda x: x.outputIndex))
        gradOutputs = tuple(inputs)
        assert len(inputs) == n
        for out1, out2 in zip(inputs, _outputs):
            assert out1.shape == out2.shape
            assert out1.dtype == out2.dtype
            assert out1.index == out2.index
        inputs = _inputs + gradOutputs
        outputs = []
        for inp in _inputs:
            name = inp.name + '_adj'
            out = FunctionOp.get_cache(name, inp)
            outputs.append(out)
        outputs = tuple(outputs)
        assert len(inputs) == len(self.args)
        assert len(outputs) == len(_inputs)
        return inputs, outputs, gradOutputs

    def _gradOutputRefs(self, gradOutputs):
        gradOutputs = list(gradOutputs)
        for index, out in enumerate(gradOutputs):
            outRef = out[out.index]
            outRef.args = (self,)
            gradOutputs[index] = outRef
        return tuple(gradOutputs)

    @classmethod
    def clear_cache(cls):
        FunctionOp._gradCache = {}

    @classmethod
    def insert_cache(cls, gradInputs):
        cache = FunctionOp._gradCache
        for out in gradInputs:
            cache[out.name] = out

    @classmethod
    def get_cache(cls, name, inp=None):
        cache = FunctionOp._gradCache
        if name in cache:
            out = cache[name]
        elif inp:
            out = Zeros(inp.shape, inp.dtype)
            out.name = name
        else:
            raise Exception("not found in cache")
        if inp:
            return out[inp.index]
        else:
            return out

class TensorFunctionOp(FunctionOp):
    _gradCache = {}
    
    def __init__(self, func, args, outputs, indices):
        assert isinstance(indices, IntegerScalar) or isinstance(indices, int)
        self.func = func
        self._init(func.name, args, outputs)
        self.indices = indices
        self.info = inspect.stack()[2:]
        
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
        n = len(self.outputs)
        args, outputs, gradOutputs = self._grad(grad)
        gradOp = TensorFunctionOp(self.func.grad, args, outputs, self.indices)
        gradInputs, gradOutputs = gradOp.outputs, gradOp._gradOutputRefs(gradOutputs)
        FunctionOp.insert_cache(gradInputs)
        gradInputs[0].args[0].info = ['grad'] + self.info
        return gradInputs + gradOutputs


class ExternalFunctionOp(FunctionOp):
    def __init__(self, name, args, outputs, empty=False):
        self._init('Function_' + name, args, outputs)
        self.empty = empty
        self.arrType = None

    def getCallString(self):
        if self.empty:
            return ''
        inp = self.args[0]
        shape = ','.join([str(x) for x in inp.shape[1:]])
        pointer = '{}<{},{}>*'.format(self.arrType, inp.dtype, shape)
        callString = 'std::vector<{}>{{'.format(pointer)
        for inp in self.args:
            callString += '({})&{},'.format(pointer, inp.name)
        callString = callString[:-1] + '}'
        return callString

    def grad(self, grad):
        n = len(self.outputs)
        args, outputs, gradOutputs = self._grad(grad)
        name = self.name[len('Function_'):] + '_grad'
        gradOp = ExternalFunctionOp(name, args, outputs, self.empty)
        gradInputs, gradOutputs = gradOp.outputs, gradOp._gradOutputRefs(gradOutputs)
        FunctionOp.insert_cache(gradInputs)
        return gradInputs + gradOutputs

class Function(object):
    _index = 0
    _module = None
    codeDir = os.path.dirname(__file__) + '/gencode/'
    codeFile = 'code.{}'.format(config.codeExt)
    funcs = []

    def __init__(self, name, inputs, outputs, grad=True):
        if config.gpu:
            self.arrType = 'gpuArrType'
        else:
            self.arrType = 'arrType'
            #self.arrType = 'gpuArrType'
        self.name = name
        self._inputs = inputs
        self._outputs = outputs
        _outputs = [x for x in self._outputs if x is not None]
        self._children = graphGetChildren(_outputs)
        if config.compile:
            self._genCode(_outputs)
        FunctionOp.clear_cache()
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
        for out in self._outputs:
            name = out.name + '_adj'
            grad = Variable(out.shape, out.dtype)
            grad.name = name
            gradients[out] = (grad,)
            gradOutputs.append(grad)
        FunctionOp.insert_cache(gradOutputs)
        gradInputs = self._diff(self._outputs, self._inputs, gradients)
        return Function(self.name + '_grad', self._inputs + gradOutputs, gradInputs, grad=False)

    def _getName(self, op):
        if isinstance(op, int):
            name = op
        else:
            name = op.name
        return name
        
    def _genCode(self, outputs):
        codeFile = open(self.codeDir + self.codeFile, 'a')
        codeFile.write('\nstatic PyObject* Function_{}(PyObject *self, PyObject *args) {{\n'.format(self.name))
        codeFile.write('\t//printf("%d %d\\n", mem.usage, mem.maxUsage);\n')
        #for out in self._outputs:
        #    memString += '{}* {}, '.format(out.dtype, out.name)
        memoryInit = {}
        for index, inp in enumerate(self._inputs):
            if isinstance(inp, IntegerScalar):
                codeFile.write('\tinteger {} = (integer) PyInt_AsLong(PyTuple_GetItem(args, {}));\n'.format(inp.name, index))
                continue
            memoryInit[inp.name] = 1
            codeFile.write('\tPyObject* Py_{} = PyTuple_GetItem(args, {});\n'.format(inp.name, index))
            shape = ','.join([str(x) for x in inp.shape[1:]])
            codeFile.write('\t{}<{}, {}> {};\n'.format(self.arrType, inp.dtype, shape, inp.name))
            if inp.static and config.gpu:
                codeFile.write('\t{}.staticVariable = true;\n'.format(inp.name))
            codeFile.write('\tgetArray((PyArrayObject*) Py_{0}, {0}, "{1}_{0}");\n'.format(inp.name, self.name))
        codeFile.write('\n')

        varChildren = {}
        for op, off in self._children.items():
            name = op.name
            if name not in varChildren:
                varChildren[name] = 0
            varChildren[name] += off
        inputNames = list(memoryInit.keys())
        outputNames = [out.name for out in outputs]

        sortedOps = graphTopologicalSort(outputs, self._children.copy())
        prevOp = Container()
        prevOp.args = []
        for op in sortedOps:

            for arg in prevOp.args:
                varName = arg.name
                assert varChildren[varName] > 0
                varChildren[varName] -= 1
                #if isinstance(arg, Variable) and varName not in outputNames and varChildren[varName] == 0:
                if isinstance(arg, Variable) and varName not in outputNames and varName not in inputNames and varChildren[varName] == 0:
                    #if varName in inputNames and ((not config.gpu) or (config.gpu and arg.static)):
                    #    continue
                    codeFile.write('\t{}.release();'.format(varName))
            
            for arg in op.args:
                varName = arg.name
                if isinstance(arg, Variable) and varName not in memoryInit:
                    shape = ','.join([str(x) for x in arg.shape[1:]])
                    arrType = '{}<{}, {}>'.format(self.arrType, arg.dtype, shape)
                    codeFile.write('\t{} {}({}, true);\n'.format(arrType, varName, self._getName(arg.shape[0]))) 
                    memoryInit[varName] = 1

            # fix garbage collection
            #for key, ref in memoryPool.items():
            #    if config.gc:
            #        codeFile.write('\t{}.destroy();\n'.format(ref.name))

            #if isinstance(op, Zeros):
            #    assert len(op.args) == 0
            #    #codeFile.write('\t// init var {}\n'.format(op.name)) 
            #    shape = ','.join([str(x) for x in op.shape[1:]])
            #    codeFile.write('\t{}<{}, {}> {}({}, true);\n'.format(self.arrType, op.dtype, shape, op.name, _getName(op.shape[0]))) 
            if isinstance(op, TensorFunctionOp):
                codeFile.write('\t/* {} */\n'.format(op.info))

                #for index, inp in enumerate(op.args[:-len(op.outputs)]):
                #    if not isinstance(inp.shape[0], int) and op.func._inputsUsed[index]:
                #        codeFile.write('\tassert({}.shape >= ({} + {}));\n'.format(inp.name, _getName(op.indices), _getName(inp.index)))
                if config.gpu:
                    name = self._getName(op.indices)
                    codeFile.write('\tif ({} > 0) {{\n'.format(name))
                    codeFile.write('\t\tinteger nBlocks = {}/GPU_THREADS_PER_BLOCK + 1;\n'.format(name))
                    codeFile.write('\t\tdim3 blocks(nBlocks / GPU_MAX_BLOCKS + 1, min(nBlocks, GPU_MAX_BLOCKS));\n')
                    codeFile.write('\t\tdim3 threads(min(GPU_THREADS_PER_BLOCK, {}));\n'.format(name))
                    #codeFile.write('\t\t{}<<<blocks, threads>>>({}, {});\n'.format(op.name, name, op.getCallString()))
                    codeFile.write('\t\t{}<<<GPU_BLOCKS_PER_GRID, GPU_THREADS_PER_BLOCK>>>({}, {});\n'.format(op.name, name, op.getCallString()))
                    codeFile.write('\t\tgpuErrorCheck(cudaPeekAtLastError());\n')
                    codeFile.write('\t}\n')
                else:
                    codeFile.write('\t{}({}, {});\n'.format(op.name, self._getName(op.indices), op.getCallString()))
            elif isinstance(op, ExternalFunctionOp):
                op.arrType = self.arrType
                codeFile.write('\t{}({});\n'.format(op.name, op.getCallString()))
            elif isinstance(op, Variable):
                pass
            else:
                raise Exception('op not recognised', op)
            prevOp = op
            
        codeFile.write('\n\tPyObject* outputs = PyTuple_New({});\n'.format(len(outputs)))
        for index, out in enumerate(outputs):
            codeFile.write('\tPyTuple_SetItem(outputs, {}, putArray({}));\n'.format(index, out.name))
        codeFile.write('\t//printf("%d %d\\n", mem.usage, mem.maxUsage);\n')
        codeFile.write('\treturn outputs;')
        codeFile.write('\n')
        codeFile.write('}\n\n')

        codeFile.close()

    def _diff(self, outputs, inputs, gradients=None):
        children = self._children.copy()
        #print children.values()
        # TODO: better handling of None, change integer grad to None
        def _diffFunc(out):
            #assert children[out] == 0
            grads = out.grad(gradients[out])
            assert len(grads) == len(out.args)
            for grad, inp in zip(grads, out.args):
                if inp not in gradients:
                    gradients[inp] = tuple()
                if not isinstance(inp, Variable):
                    gradients[inp] += (grad,)
                else:
                    gradients[inp] = (FunctionOp.get_cache(grad.name),)
                children[inp] -= 1
                if children[inp] == 0:
                    _diffFunc(inp)
        for out in outputs:
            if children[out] == 0:
                _diffFunc(out)
        #print children.values()
        #exit(1)
        #print(self.name, [len(gradients.get(inp, (None,))) for inp in inputs])
        #return [gradients.get(inp, (None,))[-1] for inp in inputs]
        return [gradients.get(inp, (None,))[0] for inp in inputs]

    def __call__(self, *args):
        func = getattr(Function._module, self.name)
        return func(*args)

    @classmethod
    def createCodeDir(self, case):
        self.codeDir = case + 'gencode/'
        if config.compile:
            if os.path.exists(self.codeDir):
                shutil.rmtree(self.codeDir)
            shutil.copytree(scriptDir + '/gencode', self.codeDir)
            Function.clean()

    @classmethod
    def clean(self):
        if not config.compile:
            return
        with open(self.codeDir + self.codeFile, 'w') as f:
            f.write('#include "code.hpp"\n')

    @classmethod
    def compile(self):
        if config.compile:
            with open(self.codeDir + self.codeFile, 'a') as f:
                f.write("PyMethodDef ExtraMethods[] = {\n")
                for name in Function.funcs:
                    f.write('\t{{"{0}", Function_{0}, METH_VARARGS, "boo"}},\n'.format(name))
                f.write("\n\t\t{NULL, NULL, 0, NULL}        /* Sentinel */\n\t};\n")
            if config.openmp:
                os.environ['WITH_OPENMP'] = '1'
            if config.matop:
                os.environ['WITH_MATOP'] = '1'
            if config.gpu:
                os.environ['WITH_GPU'] = '1'
            #if config.py3:
            #    subprocess.check_call(['make', 'python3'], cwd=self.codeDir)
            subprocess.check_call(['make'], cwd=self.codeDir)
        config.parallel.mpi.Barrier()
        sys.path.append(self.codeDir)
        import graph
        Function._module = graph

