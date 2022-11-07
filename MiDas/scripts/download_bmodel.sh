pip3 install dfn

python3 -m dfn --url http://219.142.246.77:65000/sharing/DkaMoRlRm  
mv midas_s.bmodel ../dataset/midas_s_fp32b1.bmodel


python3 -m dfn --url http://219.142.246.77:65000/sharing/H96XGZzLb
mv midas_s_int8b1.bmodel ../dataset/midas_s_int8b1.bmodel

