root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
umodel_dir=$root_dir/data/int8model

calibration_use_pb quantize  -model=$umodel_dir/flasr_a_fp32_bmnett_test_fp32.prototxt \
                            -weights=$umodel_dir/flasr_a_fp32_bmnett.fp32umodel \
                            -iterations=40 \
                            -save_test_proto=true \
                            -fpfwd_blocks="input_image_evaluate_y,input_image_evaluate_pbpr"