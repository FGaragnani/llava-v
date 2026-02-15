#!/bin/bash

sbatch scripts/v1_5/eval/cambrian.sh llava-v_s2--mean-midL-full-iti
sbatch scripts/v1_5/eval/cambrian.sh llava--only-GLAMM--mean-midL-full
sbatch scripts/v1_5/eval/cambrian.sh llava--only-GLAMM--mean-full