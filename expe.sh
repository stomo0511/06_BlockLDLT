#!/bin/bash


mm=(3072 4096 6144 8192 12288 16384 24576 32768)
bb=(128 256 512)

for j in "${bb[@]}"
  do
  for i in "${mm[@]}"
    do
      for ((k=0;k<10;k++))
      do
        ./BlockLDLT $i $j
      done
    done
  done
