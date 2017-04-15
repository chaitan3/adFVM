#!/bin/sh
DIR=$1
FILETYPE=png
NAME=$2
RATE=6
OUTPUT=$DIR/$NAME.mp4

rm -f $OUTPUT

#FILES=$(ls -r $DIR/$NAME*.$FILETYPE)
#cat $FILES | avconv -y -f image2pipe -vcodec mjpeg -i - -r $RATE  -vcodec libx264 $OUTPUT 
avconv -r $RATE -i "$NAME.%04d.$FILETYPE" -b:v 10M $OUTPUT
