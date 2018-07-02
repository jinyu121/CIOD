import json
import os

from tqdm import tqdm
from xmltodict import unparse

NUM_GROUPS = 4
START_FROM_1 = False

dataset_base = os.path.join("data", "VOCdevkitCOCO", "VOCCOCO")
dataset_dirs = {x: os.path.join(dataset_base, x) for x in ["Annotations", "ImageSets", "JPEGImages"]}
dataset_dirs['ImageSets'] = os.path.join(dataset_dirs['ImageSets'], "Main")
for k, d in dataset_dirs.items():
    os.makedirs(d, exist_ok=True)


def base_dict(filename, width, height, depth=3):
    return {
        "annotation": {
            "filename": os.path.split(filename)[-1],
            "folder": "VOCCOCO", "segmented": "0", "owner": {"name": "unknown"},
            "source": {'database': "The COCO 2017 database", 'annotation': "COCO 2017", "image": "unknown"},
            "size": {'width': width, 'height': height, "depth": depth},
            "object": []
        }
    }


def base_object(size_info, name, bbox, format="xyxy"):
    assert format in ['xyxy', 'xywh'], "Unknow format"
    if 'xyxy' == format:
        x1, y1, x2, y2 = bbox
    else:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

    width = size_info['width']
    height = size_info['height']
    offset = 0 if START_FROM_1 else 1

    x1 = max(x1, 1) - offset
    y1 = max(y1, 1) - offset
    x2 = min(x2, width) - offset
    y2 = min(y2, height) - offset

    return {
        'name': name, 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0',
        'bndbox': {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
    }


src_base = os.path.join("data", "coco", "annotations")
sets = {
    "trainval": os.path.join(src_base, "instances_train2017.json"),
    "test": os.path.join(src_base, "instances_val2017.json"),
}

cate = {x['id']: x['name'] for x in json.load(open(sets["test"]))['categories']}

for stage, filename in sets.items():
    print("Parse", filename)
    data = json.load(open(filename))

    images = {}
    for im in tqdm(data["images"], "Parse Images"):
        img = base_dict(im['coco_url'], im['width'], im['height'], 3)
        images[im["id"]] = img

    for an in tqdm(data["annotations"], "Parse Annotations"):
        ann = base_object(images[an['image_id']]['annotation']["size"],
                          cate[an['category_id']], an['bbox'], format="xywh")
        images[an['image_id']]['annotation']['object'].append(ann)

    for k, im in tqdm(images.items(), "Write Annotations"):
        unparse(im,
                open(os.path.join(dataset_dirs["Annotations"], "{}.xml".format(str(k).zfill(12))), "w"),
                full_document=False, pretty=True)

    print("Write image sets")
    with open(os.path.join(dataset_dirs["ImageSets"], "{}.txt".format(stage)), "w") as f:
        f.writelines(list(map(lambda x: str(x).zfill(12) + "\n", images.keys())))

    print("OK")
