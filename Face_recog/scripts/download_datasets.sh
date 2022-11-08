pip3 install dfn
# download pics
python3 -m dfn --url http://219.142.246.77:65000/sharing/dTvw1m9kj
# download bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/UZeq620MC

unzip bmodel.zip  
unzip data.zip
mv ./bmodel  ../dataset
mv ./data  ../dataset

rm bmodel.zip data.zip