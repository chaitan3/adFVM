#!/bin/bash
DIR=$(dirname "${BASH_SOURCE[0]}")

pytest $DIR/test_*.py

