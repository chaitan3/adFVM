import vane_profile
import numpy as np

import subprocess

import pickle
with open(sys.argv[1]) as f:
    param, base, case = pickle.load(f)
vane_profile.gen_mesh_param(param, base, case, subprocess.check_call)
