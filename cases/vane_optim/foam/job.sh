#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=talnikar@mit.edu
#SBATCH --mail-type=ALL
set -e
source /etc/profile.d/master-bin.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/sources/petsc/arch-linux2-c-opt/lib
export PYTHONPATH=$HOME/.local/lib/python.7/site-packages/:$PYTHONPATH
mpirun ~/adFVM/apps/problem.py ./vane_optim_adj.py > primal_output.log 2>primal_error.log
for i in `seq 0 39`; do
	mpirun ~/adFVM/apps/adjoint.py ./vane_optim_adj.py >> adjoint_output.log 2>>adjoint_error.log
done
