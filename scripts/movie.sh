#!/bin/sh
DIR=$1
FILETYPE=jpg
NAME=rhoaByV
RATE=6
OUTPUT=$DIR/output.mp4
FILES=$(ls -r $DIR/$NAME*.$FILETYPE)
rm -f $OUTPUT
cat $FILES | avconv -y -f image2pipe -vcodec mjpeg -i - -r $RATE  -vcodec libx264 $OUTPUT 
