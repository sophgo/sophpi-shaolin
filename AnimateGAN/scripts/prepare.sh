
echo "download pb file ....."
python3 -m dfn --url http://219.142.246.77:65000/sharing/PhIh2uhgu   

echo "Convert pb to bmodel ....."
python3 convert_pt_2_bmodel.py

echo "move bmodel ....."
mv ./compilation1684/compilation.bmodel ./animategan.bmodel  

echo "install neccessary packages ....."
cd ..
pip3 install -r requirements.txt
