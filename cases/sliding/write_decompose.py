#!/usr/bin/python2
import numpy as np

n = 20
x = np.linspace(0,0.01,n+1)
f = open('system/setSetCommands', 'w')
for i in range(0, n):
    #f.write('faceSet intersection{2} new boxToFace (0.04799 -0.2 {0}) (0.04801 0 {1})\n'.format(x[i]+0.01/200-1e-5,x[i+1]-0.01/200+1e-5, i+1))
    f.write('faceSet intersection{2} new boxToFace (0.04799 -0.2 {0}) (0.04801 0 {1})\n'.format(x[i]+5e-5,x[i+1]-5e-5, i+1))
f.close()

import re
d = 'system/decomposeParDict'
f = open(d, 'r+')
data = f.read()
f.close()
new = 'singleProcessorFaceSets (\n'
for i in range(0, n):
    new += '\t(intersection{0} {0})\n'.format(i+1)
new += ');\n'
data = re.sub(re.compile('singleProcessorFaceSets (.*?);', re.DOTALL), new, data)
f = open(d, 'w')
f.write(data)
f.close()
