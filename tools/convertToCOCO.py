import h5py
import json
from PIL import Image


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][()]])


def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr[()][i].item()][()][0][0]
                  for i in
                  range(len(attr))] if len(attr) > 1 else [attr[()][0][0]]
        attrs[key] = values
    return attrs


def h5_convert_to_coco(in_file, image_prefix):
    annotations = []
    images = []
    obj_count = 0

    for idx in range(len(in_file['digitStruct/name'])):
        impath = image_prefix + get_name(idx, in_file)
        image = Image.open(impath).convert('RGB')
        imgwidth, imgheight = image.size

        images.append(dict(
                      id=idx,
                      file_name=get_name(idx, in_file),
                      height=imgheight,
                      width=imgwidth))

        labels = get_bbox(idx, in_file)["label"]
        left = get_bbox(idx, in_file)["left"]
        top = get_bbox(idx, in_file)["top"]
        width = get_bbox(idx, in_file)["width"]
        height = get_bbox(idx, in_file)["height"]

        for img_obj_count in range(len(labels)):
            x_min, y_min, w, h = (
                left[img_obj_count], top[img_obj_count],
                width[img_obj_count], height[img_obj_count])

            if int(labels[img_obj_count]) == 10:
                labels[img_obj_count] = 0
            else:
                labels[img_obj_count] = int(labels[img_obj_count])

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=labels[img_obj_count],
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


# f = h5py.File(filename, 'r')
# print(list(f.keys()))
# print(f['table'].keys())
# print(f['table']['block0_items'])
# pd.read_hdf(filename, 'table')

mf = h5py.File('./data/digitStruct.mat', 'r')
# print(get_name(24, mf))
# print(get_bbox(24, mf))
# print(len(mf['digitStruct/name']))

image_prefix = './data/train/'
out_file = './data/test.json'

with open(out_file, 'w') as json_file:
    json.dump(h5_convert_to_coco(mf, image_prefix), json_file)

mf.close()
