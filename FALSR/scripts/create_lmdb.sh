#!/bin/bash
root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)

python3 ../tools/create_lmdb.py --imageset_rootfolder  $root_dir/data/dataset/Urban100 \
                                --resize_height 270 \
                                --resize_width 480 \
                                --scale 2 \
                                --outlmdb_rootfolder $root_dir/data/lmdb