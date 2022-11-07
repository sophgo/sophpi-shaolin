#!/bin/bash
# sh model_info.sh
root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
out_dir=$root_dir/data/fp32bmodel

python3 -m bmnett --model=$root_dir/data/models/FALSR-A.pb \
        --input_names="input_image_evaluate_y,input_image_evaluate_pbpr" \
        --shapes=[[1,270,480,1],[1,540,960,2]] \
        --output_names="test_sr_evaluator_i1_b0_g/target" \
        --net_name="falsr_a_fp32" \
        --outdir=$out_dir \
        --target=BM1684 


mv $out_dir/compilation.bmodel $out_dir/falsr_a_fp32.bmodel