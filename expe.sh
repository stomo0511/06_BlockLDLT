#!/bin/bash

mm=(4096 6144 8192 10240 12288 14336 16384 18432 20480 22528 24576 26624 28672 30720 32768)
bb=(128 256 384 512)

echo "Size, Factorize, Solve, Norm1, I.R., Norm2"

# for j in "${bb[@]}"
# do
    for i in "${mm[@]}"
    do
	./BlockLDLT $i
    done
# done
