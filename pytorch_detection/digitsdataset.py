import torch
import os
import h5py
import glob
from PIL import Image
import re


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


def convert(text):
    return int(text) if text.isdigit() else text.lower()


def alphanum_key(key):
    return [convert(c) for c in re.split('([0-9]+)', key)]


def sorted_alphanumeric(data):
    return sorted(data, key=alphanum_key)


class DigitsDataset(object):
    def __init__(self, root, transforms, mode):
        self.root = root
        self.transforms = transforms
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))
        if mode == "train":
            self.annospath = h5py.File(
                os.path.join(root, 'digitStruct.mat'), 'r')
        self.mode = mode
        if mode == "test":
            self.testimg = []
            self.testimg_id = []
            # sort the images
            imagepath = sorted_alphanumeric(
                glob.glob(f'{root}/{mode}' + '/*.png'))
            for imname in imagepath:
                self.testimg.append(imname)
                self.testimg_id.append(imname.split('/')[-1].split('.')[0])

    def __getitem__(self, idx):
        if self.mode == "train":
            # load images
            img_path = os.path.join(self.root, "train",
                                    get_name(idx, self.annospath))
            img = Image.open(img_path).convert("RGB")

            # get bounding box coordinates
            num_objs = len(get_bbox(idx, self.annospath)["label"])
            boxes = []
            left = get_bbox(idx, self.annospath)["left"]
            top = get_bbox(idx, self.annospath)["top"]
            width = get_bbox(idx, self.annospath)["width"]
            height = get_bbox(idx, self.annospath)["height"]
            for i in range(num_objs):
                xmin = left[i]
                xmax = left[i] + width[i]
                ymin = top[i]
                ymax = top[i] + height[i]
                boxes.append([xmin, ymin, xmax, ymax])

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            labels = [int(element) for element in
                      get_bbox(idx, self.annospath)["label"]]
            labels = [0 if x == 10 else x for x in labels]
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                img, target = self.transforms(img, target)
        else:
            img = Image.open(self.testimg[idx]).convert("RGB")
            target = self.testimg_id[idx]
            if self.transforms is not None:
                img = self.transforms(img)

        return img, target

    def __len__(self):
        if self.mode == "train":
            return len(self.annospath['digitStruct/name'])
        else:
            return len(self.testimg)
