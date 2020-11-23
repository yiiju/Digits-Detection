# Digits Detection

## Hardware
Ubuntu 18.04 LTS

Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

1x GeForce RTX 2080 Ti

## Set Up
### Install Dependency
All requirements is detailed in requirements.txt.

    $ pip install -r requirements.txt

### Unzip dataset
Download dataset from [SVHN dataset](https://drive.google.com/drive/folders/1lXrT_oZevViwO9Yw5iI5psVJp_plqbV7?usp=sharing).

Create /dataset directory and move the `test.zip` and `train.tar.gz` into /dataset directory.

Execute the following instruction
    
    $ sh data.sh

 - Create /data directory and unzip dataset into /data.
 - Generate `train_coco_anno.json` in /data directory.
 - Generate `fake_test.json` in /data directory.

### Using pytorch_detection

Into [pytorch_detection](/pytorch_detection) directory.

The model is Faster R-CNN with backbone ResNet50.

The details of training and test are written below.

The helper functions in [pytorch_detection/detection](/pytorch_detection/detection) directory are copy from [pytorch/vision/references/detection](https://github.com/pytorch/vision/tree/master/references/detection)

```
└── pytorch_detection 
    ├── detection ─ the helper functions from pytorch
    ├── test.py
    ├── train.py
    └── digitsdataset.py - Custom Dataset for SVHN dataset
```

### Using mmdetection
    Need to change `num_classes` in model

### Coding Style
Use PEP8 guidelines.

    $ pycodestyle *.py

## Dataset - The Street View House Numbers (SVHN) Dataset
The data directory is structured as:
```
└── data 
    ├── test ─ 13,068 test images
    ├── train ─ 33,402 training images
    ├── digitStruct.mat ─ bounding box
    ├── fake_test.json ─ fake test annotations in json format
    └── train_coco_anno.json - training annotations in json format
```

## Train
Train in PyTorch. (The root is in pytorch_detection)

    $ python3 pytorch_detection/train.py

Argument
 - `--gpuid` the ids of gpus to use
 - `--epochs` the path to store the checkpoints and config setting
 - `--batch_size` the batch size
 - `--print_freq` the frequence of print batch metric
 - `--store_dir` the directory to store checkpoints

Train in mmdetection. (The root is in the mmdetection)

    $ python3 tools/train.py configs/digits/yolov3_d53_mstrain-680_273e_digits.py --gpu-ids 8 --work-dir "work_dirs/yolov3_d53_mstrain-680_273e_digits_epoch50"

Argument
 - `--gpu-ids` the ids of gpus to use
 - `--work-dir` the path to store the checkpoints and config setting

## Inference
Test in PyTorch. (The root is in pytorch_detection)

    $ python3 pytorch_detection/test.py

Argument
 - `--gpuid` the ids of gpus to use
 - `--batch_size` the batch size
 - `--checkpoint` the path of load checkpoint model
 - `--pname` the path to store the result

Test in mmdetection. (The root is in the mmdetection)

    $ python tools/test.py configs/digits/yolov3_d53_mstrain-680_273e_digits.py work_dirs/yolov3_d53_mstrain-680_273e_digits/latest.pth --format-only --options "jsonfile_prefix=../result/yolo/yolov3_epoch65"
    $ cd ..
    $ python GenAnsJson.py --inpath "result/yolo/yolov3_epoch65.bbox.json" --outpath "result/mmdetection/upload/upload_yolov3_epoch65.json"

Argument
 - `--inpath` the path of JSON file generated by mmdetection
 - `--outpath` the path of wanted JSON file 

## Citation
```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```