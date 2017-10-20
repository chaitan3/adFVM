#!/bin/sh
CASE=$1
CASEDIR=$(readlink -f $CASE/par-64)
SERVER=voyager.mit.edu
cp ~/adFVM/templates/vane_optim_adj.py $CASEDIR
rsync -aRv $CASEDIR/./* $SERVER:$CASEDIR
rsync -aRv job.sh $SERVER:$CASEDIR
ssh $SERVER "cd $CASEDIR && ./job.sh"

