#!/bin/sh
DIR=$1
RATE=2
OUTPUT=$DIR/output.mp4
rm $OUTPUT
FILES=$(ls -r $DIR/*.png)
echo $FILES
cat $FILES | \
ffmpeg -f image2pipe -r $RATE -i - \
    -vcodec libx264 $OUTPUT 
