import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

import os
import json
import argparse
import numpy as np

import detection.transforms as T
import detection.utils as utils
from digitsdataset import DigitsDataset


def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                      std=[0.5, 0.5, 0.5]),
    ])
    return transform


def gen_predict_json(finallist, predictions):
    for i in range(len(predictions)):
        bbox = []
        for box in predictions[i]['boxes'].tolist():
            box = [int(element) for element in box]
            order = [1, 0, 3, 2]
            box = [box[i] for i in order]
            bbox.append(box)
        labels = [10 if x == 0 else x
                  for x in predictions[i]['labels'].tolist()]
        finallist.append(dict(bbox=bbox,
                              label=labels,
                              score=predictions[i]['scores'].tolist()))
    return finallist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuid", type=int, nargs='+', default=8,
                        help="ids of gpus to use")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="size of each image batch")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/epoch10/1.pth",
                        help="load checkpoint model")
    parser.add_argument("--pname", type=str, default="1.json",
                        help="path to store the result")
    opt = parser.parse_args()

    # train on the GPU or on the CPU, if a GPU is not available
    gpuid = str(opt.gpuid)
    if isinstance(opt.gpuid, list):
        gpuid = ""
        for i in opt.gpuid:
            gpuid = gpuid + str(i) + ", "
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        torch.device('cpu')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
    num_classes = 10
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(opt.checkpoint))
    model.to(device)
    dataset = DigitsDataset('./data', get_transform(), mode="test")
    testLoader = torch.utils.data.DataLoader(
                    dataset, batch_size=opt.batch_size,
                    shuffle=False, num_workers=4,
                    collate_fn=utils.collate_fn)

    # For inference
    model.eval()
    with torch.no_grad():
        listdist = []
        for data in testLoader:
            images, imageid = data
            # images = [np.asarray(img) for img in images]
            # print(np.asarray(images))
            images = list(image.to(device) for image in images)
            predictions = model(images)
            listdist = gen_predict_json(listdist, predictions)

    out_file = "result/" + opt.pname
    with open(out_file, 'w') as json_file:
        json.dump(listdist, json_file)
