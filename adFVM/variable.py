
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

    def __getitem__(self, index):
        #print self.index
        #assert self.index == 0
        var = self.getReference()
        var.index = index
        return var

    def getReference(self):
        var = Variable(self.shape)
        var.args = (self,)
        var.name = self.name
        return var

class Zeros(Variable):
    pass

class TensorFunctionOp(object):
    def __init__(self, func, args, outputs, indices):
        self.func = func
        self.name = func.name
        n = len(self.func._inputTensors)
        self.indices = indices
        args = args + outputs
        self.args = args
        self.outputs = [x.getReference() for x in outputs]
        for out in self.outputs:
            out.args = (self,)

class Function(object):
    _index = 0
    _module = None
    codeDir = os.path.dirname(__file__) + '/gencode/'

    def __init__(self, name, inputs, outputs):
        self.name = name
        self._inputs = inputs
        self._outputs = outputs
        self._children = graphGetChildren(outputs)
        self._genCode()

    def _genCode(self):
        codeFile = open(self.codeDir + 'code.cpp', 'a')
        memString = '' 
        for inp in self._inputs:
            memString += 'const {}* {}, '.format(inp.dtype, inp.name)
        for out in self._outputs:
            memString += '{}* {}, '.format(out.dtype, out.name)
        codeFile.write('\nvoid Function_{}({}) {}\n'.format(self.name, memString[:-2], '{\n'))

        sortedOps = graphTopologicalSort(self._outputs, self._children.copy())
        for op in sortedOps:
            if isinstance(op, Zeros):
                print op.name, op.args
                assert len(op.args) == 0
                codeFile.write('// init var {}\n'.format(op.name)) 
            elif isinstance(op, Variable):
                if len(op.args) == 0:
                    codeFile.write('// input var {}\n'.format(op.name)) 
                #else:
                #    codeFile.write('// dependant var {}\n'.format(op.name)) 
            elif isinstance(op, TensorFunctionOp):
                print op
            elif isinstance(op, ConstScalar):
                print op
            else:
                raise Exception('op not recognised', op)

        codeFile.write('}')

        codeFile.close()
        exit(1)

    @classmethod
    def createCodeDir(self, case):
        self.codeDir = case + 'gencode/'
        if config.compile:
            assert not os.path.exists(self.codeDir)
            shutil.copytree(scriptDir + '/gencode', self.codeDir)

    @classmethod
    def clean(self):
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
                f.write('\n\n' + self.extraCode)
            subprocess.check_call(['make'], cwd=self.codeDir)
        parallel.mpi.Barrier()
        sys.path.append(self.codeDir)
        import interface as mod
        Function._module = mod

