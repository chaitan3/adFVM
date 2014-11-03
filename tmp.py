from mesh import Mesh
from field import CellField

mesh = Mesh('tests/cylinder')

class empty: pass
solver = empty()
solver.mesh = mesh
CellField.setSolver(solver)

U = CellField.read('U', 2.0)
