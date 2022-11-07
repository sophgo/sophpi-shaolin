#!/bin/bash

root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)

python3 $root_dir/tools/int8u_to_bmodel.py

mv $root_dir/data/int8model/compilation.bmodel $root_dir/data/int8model/falsr_a_int8.bmodel