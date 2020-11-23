# Digits Detection - tools

## Project Files
`tools/convertToCOCO.py`

Generate training annotations from `digitStruct.mat` into coco dataset format `train_coco_anno.json`.

`tools/GenAnsJson.py`

Generate test annotations from the output in mmdetection to wanted format. (JSON to JSON)

`tools/fakeJson.py`

Generate fake test annotation in wanted JSON format.

`tools/AnalysisData.py`

Calculate mean and standard deviation of the training data.