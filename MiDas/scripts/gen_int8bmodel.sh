cd ../python 
python3 make_lmdb.py 

cd ../scripts 

python3 -m ufw.tools.pt_to_umodel \
    -m midas_s_192_256_jit.pt \
    -d compilation \
    -s "[1,3,192,256]" \
    -D ../dataset/lmdb \
    --cmp

mv ./compilation/*net* ./

calibration_use_pb  quantize -model=midas_s_192_256_jit_bmnetp_test_fp32.prototxt -weights=midas_s_192_256_jit_bmnetp.fp32umodel -iterations=10  -fpfwd_outputs="< 12 >17,< 12 >23,< 12 >28"

bmnetu --model=./midas_s_192_256_jit_bmnetp_deploy_int8_unique_top.prototxt \
        --weight=./midas_s_192_256_jit_bmnetp.int8umodel \
        --outdir=./int8bmodel \
        --shapes=[1,3,192,256] \
        --cmp false

mv ./int8bmodel/compilation.bmodel ../dataset/midas_s_int8b1.bmodel 