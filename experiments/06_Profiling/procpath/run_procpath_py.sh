#!/bin/bash

python run_pypsupertime.py -f ~/data/mem_benchmark_1146x15778.h5ad &
C1=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_1719x15778.h5ad &
C2=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_2292x15778.h5ad &
C3=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_2865x15778.h5ad &
C4=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_3438x15778.h5ad &
C5=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_4011x15778.h5ad &
C6=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_4584x15778.h5ad &
C7=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_5157x15778.h5ad &
C8=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_5730x15778.h5ad &
C9=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_6303x15778.h5ad &
C10=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_6876x15778.h5ad &
C11=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_7449x15778.h5ad &
C12=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_8022x15778.h5ad &
C13=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_8595x15778.h5ad &
C14=$!

python run_pypsupertime.py -f ~/data/mem_benchmark_9168x15778.h5ad &
C15=$!


procpath record -i 1 -v 30 -d memtest_py.sqlite -p "$C1,$C2,$C3,$C4,$C5,$C6,$C7,$C8,$C9,$C10,$C11,$C12,$C13,$C14,$C15"

