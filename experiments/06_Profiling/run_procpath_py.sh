#!/bin/bash

python test.py &
PID=$!
echo "$PID"

procpath record -i1 -r 120 -d mem_results/pypsupertime.sqlite -p $PID

