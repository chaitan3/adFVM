#!/bin/bash
DIR=$(dirname "${BASH_SOURCE[0]}")

pytest $DIR/test_op.py $DIR/test_interp.py $DIR/test_parallel.py $DIR/test_field.py
