import re
import json
import glob
from PIL import Image


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def gen_coco_format():
    annotations = []
    images = []
    obj_count = 0

    imgspath = []
    imgname = []
    imagepath = sorted_alphanumeric(
                glob.glob("./data/test" + '/*.png'))
    for imname in imagepath:
        imgspath.append(imname)
        imgname.append(imname.split('/')[-1])

    for idx in range(len(imgname)):
        image = Image.open(imgspath[idx]).convert('RGB')
        imgwidth, imgheight = image.size

        images.append(dict(
                      id=idx,
                      file_name=imgname[idx],
                      height=imgheight,
                      width=imgwidth))

        x_min, y_min, w, h = ( 0, 0, 0, 0)
        data_anno = dict(
            image_id=idx,
            id=obj_count,
            category_id= 0,
            bbox=[x_min, y_min, w, h],
            area=w * h,
            segmentation=[],
            iscrowd=0)

        annotations.append(data_anno)
        obj_count += 1

    coco_format_json = dict(images=images,
                            annotations=annotations,
                            categories=[{'id': 0, 'name': '10'},
                                        {'id': 1, 'name': '1'},
                                        {'id': 2, 'name': '2'},
                                        {'id': 3, 'name': '3'},
                                        {'id': 4, 'name': '4'},
                                        {'id': 5, 'name': '5'},
                                        {'id': 6, 'name': '6'},
                                        {'id': 7, 'name': '7'},
                                        {'id': 8, 'name': '8'},
                                        {'id': 9, 'name': '9'}])
    return coco_format_json

out_file = './data/fake_test.json'

with open(out_file, 'w') as json_file:
    json.dump(gen_coco_format(), json_file)
