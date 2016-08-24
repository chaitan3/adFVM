#!/bin/sh
set -e

#python ~/adFVM/scripts/field/map_fields.py ../laminar/ ./ 2.0 2.0

./write_decompose.py
setSet -batch system/setSetCommands
decomposePar -time 2

