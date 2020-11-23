import json
import argparse
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, default="../result/yolo/yolov3_try.bbox.json",
                        help="path to store the result")
    parser.add_argument("--outpath", type=str, default="../result/yolo/upload/new.json",
                        help="path to store the result")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.inpath) as f:
        data = json.load(f)
        out_list = convert_to_uplaod_format(data)

    with open(args.outpath, 'w') as json_file:
        json.dump(out_list, json_file)


def convert_to_uplaod_format(in_json):
    bbox = []
    labels = []
    scores = []
    out_list = []
    prename = ""
    count = 0
    for idx in range(len(in_json)):
        imgname = in_json[idx]['image_id']
        left = in_json[idx]['bbox'][0]
        top = in_json[idx]['bbox'][1]
        width = in_json[idx]['bbox'][2]
        height = in_json[idx]['bbox'][3]
        bottom = top + height
        right = left + width
        box = [int(top), int(left), int(bottom), int(right)]
        label = 10 if in_json[idx]['category_id'] == 0 else in_json[idx]['category_id']
        if imgname == prename or idx == 0:
            bbox.append(box)
            labels.append(label)
            scores.append(in_json[idx]['score'])
        else:
            out_list.append(dict(bbox=bbox,
                              label=labels,
                              score=scores))
            count = count + 1
            if imgname-prename > 1:
                empty = imgname - prename - 1
                print(imgname, prename)
                while empty:
                    empty = empty - 1
                    out_list.append(dict(bbox=[],
                                label=[],
                                score=[]))
                    count = count + 1
            bbox = [box]
            labels = [label]
            scores = [in_json[idx]['score']]
        if idx == len(in_json)-1:
            out_list.append(dict(bbox=bbox,
                              label=labels,
                              score=scores))
            count = count + 1
        prename = imgname
    print(count)    
    return out_list


if __name__ == '__main__':
    main()