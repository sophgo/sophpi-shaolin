#!/bin/bash
root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
build_dir=$root_dir/build
# src_model_file=$build_dir/data/models/"FALSR-A.pb"
src_model_name=`basename ${src_model_file}`
lmdb_dst_dir="${build_dir}/data/lmdb/"

input_width_1=480
input_height_1=270

input_width_2=960
input_height_2=540


function check_file()
{
    if [ ! -f $1 ]; then
        echo "$1 not exist."
        exit 1
    fi
}

function check_dir()
{
    if [ ! -d $1 ]; then
        echo "$1 not exist."
        exit 1
    fi
}