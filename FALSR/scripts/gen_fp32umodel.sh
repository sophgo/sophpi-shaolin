#!/bin/bash
root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
out_dir=$root_dir/data/int8model

python3 -m ufw.tools.tf_to_umodel  --dataset=''$root_dir'/data/lmdb/data_1.lmdb;'$root_dir'/data/lmdb/data_2.lmdb' \
                                 --model=$root_dir/data/models/FALSR-A.pb \
                                 --input_names='input_image_evaluate_y,input_image_evaluate_pbpr' \
                                 --output_names='test_sr_evaluator_i1_b0_g/target' \
                                 --shapes='[1,270,480,1],[1,540,960,2]' \
                                 --net_name='flasr_a_fp32' \
                                 --dir=$out_dir