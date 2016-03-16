#!/bin/sh
DIR=$1
RATE=6
OUTPUT=$DIR/output.mp4
rm $OUTPUT
cat $(ls -t $DIR/*.png) | \
ffmpeg -f image2pipe -r $RATE -i - \
    -vcodec libx264 $OUTPUT
