#!/bin/bash
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
REMOTE_GPU_ADDRESS=10.68.191.91 LD_PRELOAD=./cricket-client.so ./my_app


