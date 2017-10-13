#!/bin/bash
nvcc vectorTruncat.cu -o exe/vectorTruncat -O3 -std=c++11 -D_DEBUG\
&& sleep 3s && \
CUDA_VISIBLE_DEVICES=1 ./exe/vectorTruncat
