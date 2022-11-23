#!/bin/bash
sleep 6h
python ../sweep_dense_position_random.py >/home1/cxy/A-SPRF/logs/random_sweep_ance_20_0921.out 2>&1
wait
echo "finish!"