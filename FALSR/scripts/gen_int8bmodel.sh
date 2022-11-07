root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
src_model=$root_dir/data/models/FALSR-A.pb

sh ./create_lmdb.sh
sh ./gen_fp32umodel.sh
sh ./gen_int8umodel.sh
sh ./int8u2bmodel.sh
