#!/bin/bash

sbatch scripts/v1_5/eval/cambrian.sh llava-v_s2--mean
sbatch scripts/v1_5/eval/cambrian.sh llava-v_s2--last
sbatch scripts/v1_5/eval/cambrian.sh llava-v_s2--last-midL