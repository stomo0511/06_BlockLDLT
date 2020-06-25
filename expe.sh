#!/bin/bash


#mm=(3072 4096 6144 8192 12288 16384 24576 32768)
mm=(24576)

for i in "${mm[@]}"
  do
    for ((k=0;k<10;k++))
    do
      ./BlockLDLT $i
    done
  done
