#!/bin/bash
CASE=$1
SERVER=voyager
SERVERDIR=adFVM/cases/vane_optim/optim/$CASE/par-64
set -e
scp $SERVER:$SERVERDIR/processor0/{ener*,sens*,time*} $CASE/processor0
ssh $SERVER "cat $SERVERDIR/processor0/objective.txt" >> $CASE/processor0/objective.txt
ssh $SERVER "rm -rf $SERVERDIR"
