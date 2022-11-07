python3 -m bmnetp --model=./midas_s_192_256_jit.pt \
        --shapes="[1,3,192,256]" \
        --target="BM1684"

mv ./compilation/compilation.bmodel ../dataset/midas_s_fp32b1.bmodel  
