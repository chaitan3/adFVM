#!/bin/sh
CASE=$1
#CASE=$1/processor*/
rm -rf $CASE/{*.txt,*.pkl}
rm -rf $CASE/{0.*,1.*,2.*}
