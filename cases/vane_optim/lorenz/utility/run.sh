#!/bin/sh

for i in `seq 0 999`; do
    ./grad_utility.py >> output_$i.log
    ./grad_utility.py grad >> output_grad_$i.log
done
