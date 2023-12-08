#!/bin/bash

./test.R &
PID=$!
echo "$PID"

procpath record -i1 -r 120 -d mem_results/rpsupertime.sqlite -p $PID

