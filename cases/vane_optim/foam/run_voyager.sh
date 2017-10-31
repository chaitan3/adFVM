#!/bin/sh
CASE=$(readlink -f $1)
BASE=$(basename $CASE)
PAR=par-64
CASEDIR=$CASE/$PAR
# move grad
for i in `seq 0 3`; do
    GRAD=grad$i
    mkdir -p $CASEDIR/$PAR/$GRAD
    mv $CASE/$GRAD/$PAR/processor* $CASEDIR/$PAR/$GRAD
done

# redecompose
TIME=3.001
SERVER2=kolmogov
SERVER2DIR=adFVM/cases/vane_optim/foam/laminar/$BASE
rsync -aRv $CASE/./processor* $SERVER2:$SERVER2DIR
ssh $SERVER2 "of && cd $SERVER2DIR && reconstructPar -time $TIME && $HOME/adFVM/scripts/decompose.sh ./ 64 -time $TIME"
ssh $SERVER2 "cd $SERVERDIR && mkdir $PAR && mv processor* $PAR"
rsync -aRv $SERVER2:$SERVERDIR/$PAR/./processor* $CASEDIR

# transfer
SERVERDIR=/master/$CASEDIR
SERVER=voyager
ssh $SERVER mkdir -p $SERVERDIR
rsync -aRv $CASEDIR/./* $SERVER:$SERVERDIR
rsync -aRv job.sh $SERVER:$SERVERDIR
ssh $SERVER "cd $SERVERDIR && sbatch ./job.sh"

