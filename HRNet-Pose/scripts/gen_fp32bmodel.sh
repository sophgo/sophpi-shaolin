python3 -m bmneto --model=./mmpose_256_192.onnx --shapes="[1,3,256,192]" --target="BM1684"

mv ./compilation/compilation.bmodel ../dataset/mmpose_fp32.bmodel