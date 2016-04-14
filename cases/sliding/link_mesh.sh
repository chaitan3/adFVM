#!/bin/sh
PWD=`pwd`
for DIR in processor*; do
	rm -f $DIR/2/polyMesh
	ln -s $PWD/$DIR/constant/polyMesh $PWD/$DIR/2/polyMesh
done
