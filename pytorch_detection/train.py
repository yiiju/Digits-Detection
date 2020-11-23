import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.distributed as dist

import os
import h5py
import numpy as np
import argparse
from PIL import Image

from detection.engine import train_one_epoch, evaluate
import detection.utils as utils
import detection.transforms as T

from digitsdataset import DigitsDataset


# def setup(rank, world_size):
#     # prepare the environment of process groups
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize process group
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)

#     # assign the seed to make sure the process will start in the same setting
#     torch.manual_seed(42)


# def cleanup():
#     # clean process groups
#     dist.destroy_process_group()


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuid", type=int, nargs='+', default=8,
                        help="ids of gpus to use")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="size of each image batch")
    parser.add_argument("--print_freq", type=int, default=8338,
                        help="frequence of print batch metric")
    parser.add_argument("--store_dir", type=str, default="new",
                        help="directory to store checkpoints")
    opt = parser.parse_args()

    # Create checkpoint directory
    if not os.path.exists("./checkpoints/" + opt.store_dir):
        os.mkdir("./checkpoints/" + opt.store_dir)
        print("Directory ", "./checkpoints/" + opt.store_dir, " Created")

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

    num_classes = opt.epochs
    # use our dataset and defined transformations
    dataset = DigitsDataset('./data', get_transform(train=True), mode="train")
    dataset_test = DigitsDataset('./data',
                                 get_transform(train=False), mode="train")

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=opt.batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)

    num_classes = 10
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # move model to the right device
    # setup(0, 1)
    # n = torch.cuda.device_count() // 1
    # device_ids = list(range(0 * n, (0 + 1) * n))
    # print(device)

    model.to(device)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    # construct an optimizer
    print(model)
    for p in model.parameters():
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = opt.epochs

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=opt.print_freq)
        # update the learning rate
        lr_scheduler.step()

        # cleanup()

        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        path = "./checkpoints/" + opt.store_dir + "/" + str(epoch + 1) + ".pth"
        # store the model
        torch.save(model.state_dict(), path)
