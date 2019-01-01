#!/bin/bash
DIR=$(dirname "${BASH_SOURCE[0]}")
source /opt/openfoam6/etc/bashrc

cd $DIR && pytest test_op.py test_interp.py test_parallel.py test_field.py
