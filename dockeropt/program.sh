#!/bin/bash

export JULIA_NUM_THREADS=4

#julia /app/main.jl >programOPT.log 
julia main.jl #>programOPT.log 
