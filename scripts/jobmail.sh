#!/bin/bash
OUTPUT=output.log
ERROR=error.log

CMD=$*
HOSTNAME=$(hostname)
EMAIL=chaitukca@gmail.com

echo 'Running ' $CMD
echo
START=$(date)
$CMD > >(tee $OUTPUT) 2>$ERROR
#$CMD > >(tee output.log) 2>(tee  error.log >&2)
END=$(date)

CSTART=$(echo $START| tr ":" .)
SUBJECT="$USER@$HOSTNAME $CSTART"

echo "Command: $CMD
End time: $END

Output last 10 lines: 
$(tail -n 10 $OUTPUT)

Error last 10 lines:
$(tail -n 10 $ERROR)

" | mail -s "$SUBJECT" $EMAIL

echo
echo "Done"
