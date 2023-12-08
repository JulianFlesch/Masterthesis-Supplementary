#!/bin/bash

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_1146x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C1=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_1719x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C2=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_2292x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C3=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_2865x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C4=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_3438x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C5=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_4011x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C6=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_4584x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C7=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_5157x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C8=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_5730x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C9=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_6303x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C10=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_6876x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C11=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_7449x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C12=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_8022x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C13=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_8595x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C14=$!

R -e 'sce <- zellkonverter::readH5AD("~/data/mem_benchmark_9168x15778.h5ad"); psupertime::psupertime(sce, sce$Ordinal_Time_Labels, sel_genes="all", scale=FALSE)' &
C15=$!

procpath record -i 1 -v 30 -d memtest_r.sqlite -p "$C1,$C2,$C3,$C4,$C5,$C6,$C7,$C8,$C9,$C10,$C11,$C12,$C13,$C14,$C15"

