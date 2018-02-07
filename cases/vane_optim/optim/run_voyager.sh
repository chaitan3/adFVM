#!/bin/bash
CASE=$(readlink -f $1)
BASE=$(basename $CASE)
PAR=par-64
CASEDIR=$CASE/$PAR
set -e
# move grad
for i in `seq 0 3`; do
    GRAD=grad$i
    mkdir -p $CASEDIR/$GRAD
    mv $CASE/$GRAD/$PAR/processor* $CASEDIR/$GRAD
done

# redecompose
TIME=3.00100000000
SERVER2=kolmogorov
SERVER2DIR=adFVM/cases/vane_optim/foam/laminar/$BASE
rsync -aRv $CASE/./processor*/{constant,$TIME} $SERVER2:$SERVER2DIR
ssh $SERVER2 /bin/bash << EOF
  set -e
  source /opt/openfoam240/etc/bashrc
  cd $SERVER2DIR 
  reconstructPar -time $TIME
  rm -rf processor*
  $HOME/adFVM/scripts/decompose.sh ./ 64 -time $TIME
EOF
rsync -aRv $SERVER2:$SERVER2DIR/$PAR/./processor* $CASEDIR

# transfer
SERVERDIR=/master/$CASEDIR
SERVER=voyager
ssh $SERVER mkdir -p $SERVERDIR
rsync -aRv $CASEDIR/./* $SERVER:$SERVERDIR
rsync -aRv job.sh $SERVER:$SERVERDIR
ssh $SERVER "cd $SERVERDIR && sbatch ./job.sh"

