import json
import os

from tqdm import tqdm
from xmltodict import unparse

# BBOX_OFFSET: Switch between 0-based and 1-based bbox
# The COCO dataset is in 0-based format, while the VOC is 1-based
# To keep 0-based, set it to 0. To make 1-based annotation, set it to 1.
BBOX_OFFSET = 1

src_base = os.path.join("data", "coco", "annotations")
dst_base = os.path.join("data", "VOCdevkitCOCO", "VOCCOCO")

dst_dirs = {x: os.path.join(dst_base, x) for x in ["Annotations", "ImageSets", "JPEGImages"]}
dst_dirs['ImageSets'] = os.path.join(dst_dirs['ImageSets'], "Main")
for k, d in dst_dirs.items():
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


def base_object(size_info, name, bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    width = size_info['width']
    height = size_info['height']

    x1 = max(x1, 0) + BBOX_OFFSET
    y1 = max(y1, 0) + BBOX_OFFSET
    x2 = min(x2, width - 1) + BBOX_OFFSET
    y2 = min(y2, height - 1) + BBOX_OFFSET

    return {
        'name': name, 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0',
        'bndbox': {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
    }


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
        ann = base_object(images[an['image_id']]['annotation']["size"], cate[an['category_id']], an['bbox'])
        images[an['image_id']]['annotation']['object'].append(ann)

    for k, im in tqdm(images.items(), "Write Annotations"):
        im['annotation']['object'] = im['annotation']['object'] or [None]
        unparse(im,
                open(os.path.join(dst_dirs["Annotations"], "{}.xml".format(str(k).zfill(12))), "w"),
                full_document=False, pretty=True)

    print("Write image sets")
    with open(os.path.join(dst_dirs["ImageSets"], "{}.txt".format(stage)), "w") as f:
        f.writelines(list(map(lambda x: str(x).zfill(12) + "\n", images.keys())))

    print("OK")
