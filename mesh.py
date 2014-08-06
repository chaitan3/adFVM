import re
import numpy as np

class Mesh:
    def __init__(self, caseDir):
        meshDir = caseDir + '/constant/polyMesh/'
        self.faces = self.read(meshDir + 'faces', int)
        self.points = self.read(meshDir + 'points', float)
        self.owner = self.read(meshDir + 'owner', int)
        self.neighbour = self.read(meshDir + 'neighbour', int)
        self.normals = self.getNormals()
        pass

    def read(self, foamFile, dtype):
        lines = open(foamFile).readlines()
        first, last = 0, -1
        while lines[first][0] != '(': 
            first += 1
        while lines[last][0] != ')': 
            last -= 1
        f = lambda x: filter(None, re.split('[ ()]+', x)[:-1])
        return np.array(map(f, lines[first + 1:last]), dtype)

    def getNormals(self):
        #correct direcion?
        v1 = self.points[self.faces[:,1]]-self.points[self.faces[:,2]]
        v2 = self.points[self.faces[:,2]]-self.points[self.faces[:,3]]
        normals = np.cross(v1, v2)
        return normals/ np.linalg.norm(normals, axis=1).reshape(-1,1)
