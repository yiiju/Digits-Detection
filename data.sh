mkdir data
tar zxvf ./dataset/train.tar.gz -C ./data
unzip ./dataset/test.zip -d ./data
mv data/train/digitStruct.mat data/
mv data/train/see_bboxes.m data/
python3 tools/convertToCOCO.py
python3 tools/fakeJson.py