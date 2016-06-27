#!/bin/sh
DIR=$1
FILETYPE=jpg
NAME=yplus
RATE=2
OUTPUT=$DIR/output.mp4
rm -f $OUTPUT
FILES=$(ls -r $DIR/$NAME*.$FILETYPE)
cat $FILES | \
avconv -f image2pipe -r $RATE -i - \
    -vcodec libx264 $OUTPUT 
