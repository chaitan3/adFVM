#!/bin/sh
CASE=$1
CASEDIR=$(readlink -f $CASE/par-64)
SERVERDIR=/master/$CASEDIR
SERVER=voyager
cp ~/adFVM/templates/vane_optim_adj.py $CASEDIR
ssh $SERVER mkdir -p $SERVERDIR
rsync -aRv $CASEDIR/./* $SERVER:$SERVERDIR
rsync -aRv job.sh $SERVER:$SERVERDIR
ssh $SERVER "cd $SERVERDIR && sbatch ./job.sh"

