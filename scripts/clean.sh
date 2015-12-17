#!/bin/sh
CASE=$1
#CASE=$1/processor*/
echo rm -rf $CASE/{*.txt,*.pkl}
echo rm -rf $CASE/{0.*,1.*,2.*}
