from config import ad
import config
from parallel import pprint

import numpy as np

from pyRCF import RCF
from field import IOField, CellField
from op import div, grad
from interp import central, TVD_dual


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('time', nargs='+', type=float)
user = parser.parse_args(config.args)

names = ['divU', 'gradc']
dimensions = [(1,), (3,)]

solver = RCF(user.case)
mesh = solver.mesh
paddedMesh = mesh.paddedMesh

def createComputer():
    SF = ad.matrix()
    PSF = solver.padField(SF)
    pP, UP, TP = solver.unstackFields(PSF, CellField)
    U = CellField.getOrigField(UP)
    T = CellField.getOrigField(TP)
    p = CellField.getOrigField(pP)

    #divU
    gradU = grad(central(UP, paddedMesh), ghost=True)
    ULF, URF = TVD_dual(U, gradU)
    UF = 0.5*(ULF + URF)
    divU = div(UF.dotN(), ghost=True).field

    #speed of sound
    cP = (solver.gamma*TP*solver.R).sqrt()
    gradc = grad(central(cP, paddedMesh), ghost=True).field

    computer = solver.function([SF], [divU, gradc], 'compute')
    return computer

for index, time in enumerate(user.time):
    rho, rhoU, rhoE = solver.initFields(time)
    U, T, p = solver.U, solver.T, solver.p
    SF = solver.stackFields([p, U, T], np)
    if index == 0:
        computer = createComputer()
    outputs = computer(SF)
    for field, name, dim in zip(outputs, names, dimensions):
        IO = IOField(name, field, dim)
        IO.write(time)

    # non theano outputs
    # 
    rhoa = IOField.read('rhoa', mesh, time)
    rhoaByV = p.field*0
    nInternalCells = mesh.origMesh.nInternalCells
    rhoaByV[:nInternalCells] = rhoa.field[:nInternalCells]/mesh.origMesh.volumes
    rhoaByV = IOField('rhoaByV', rhoaByV, (1,))
    rhoaByV.write(time)

    pprint()
