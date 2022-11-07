cd ../python 
python3 make_lmdb.py

cd ../scripts

# source model 2  fp32 umodel 

python3 -m ufw.tools.on_to_umodel \
    -m mmpose_256_192.onnx\
    -d compilation \
    -s "[1,3,256,192]" \
    -D ../dataset/lmdb/ \
    --cmp

mv ./compilation/*net* ./

calibration_use_pb  quantize -model=mmpose_256_192_bmneto_test_fp32.prototxt -weights=mmpose_256_192_bmneto.fp32umodel -iterations=100  -fpfwd_outputs="3719"

bmnetu --model=./mmpose_256_192_bmneto_deploy_int8_unique_top.prototxt \
       --weight=./mmpose_256_192_bmneto.int8umodel \
       --outdir=./int8bmodel \
       --shapes=[1,3,256,192] \
       --cmp false

mv ./int8bmodel/compilation.bmodel ../dataset/mmpose_int8b1.bmodel
