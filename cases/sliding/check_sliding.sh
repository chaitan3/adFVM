#!/bin/sh
for DIR in `seq 1 20`; do
    echo processor$DIR
    checkMesh -case processor$DIR -constant |grep intersection
done
