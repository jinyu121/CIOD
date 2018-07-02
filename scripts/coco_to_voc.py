import json
import os
from copy import deepcopy

from tqdm import tqdm
from xmltodict import unparse

NUM_GROUPS = 4

dataset_base = os.path.join("data", "VOCdevkitCOCO", "VOCCOCO")
dataset_dirs = {x: os.path.join(dataset_base, x) for x in ["Annotations", "ImageSets", "JPEGImages"]}
dataset_dirs['ImageSets'] = os.path.join(dataset_dirs['ImageSets'], "Main")
for k, d in dataset_dirs.items():
    os.makedirs(d, exist_ok=True)

base_dict = {
    "annotation": {
        "filename": "",
        "folder": "VOCCOCO", "segmented": "0", "owner": {"name": "unknown"},
        "source": {'database': "The COCO 2017 database", 'annotation': "COCO 2017", "image": "unknown"},
        "size": {'width': "", 'height': "", "depth": ""},
        "object": []
    }
}

base_object = {
    'name': '', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0',
    'bndbox': {'xmin': '', 'ymin': '', 'xmax': '', 'ymax': ''}
}

cate = {x['id']: x['name'] for x in
        json.load(open(os.path.join("data", "coco", "annotations", "instances_val2017.json")))['categories']}

for sets in ['trainval', 'test']:
    for group in range(NUM_GROUPS):
        for ifa in ["", "a"]:
            filename = os.path.join("data", "coco", "annotations", "{}Step{}{}".format(sets, group, ifa))

            print("Loading", filename, "...")
            data = json.load(open("{}.json".format(filename)))

            images = {}
            for im in tqdm(data["images"], "Parse Images"):
                img = deepcopy(base_dict)
                img['annotation']["filename"] = os.path.split(im['coco_url'])[-1]
                img['annotation']["size"] = {'width': str(im['width']), 'height': str(im['height']), "depth": "3"}
                images[im["id"]] = img

            for an in tqdm(data["annotations"], "Parse Annotations"):
                ann = deepcopy(base_object)
                ann['name'] = cate[an['category_id']]
                ann["bndbox"]["xmin"] = int(an['bbox'][0])
                ann["bndbox"]["ymin"] = int(an['bbox'][1])
                ann["bndbox"]["xmax"] = int(an['bbox'][0] + an['bbox'][2])
                ann["bndbox"]["ymax"] = int(an['bbox'][1] + an['bbox'][3])
                images[an['image_id']]['annotation']['object'].append(deepcopy(ann))

            for k, im in tqdm(images.items(), "Write Annotations"):
                unparse(im,
                        open(os.path.join(dataset_dirs["Annotations"], "{}.xml".format(str(k).zfill(12))), "w"),
                        full_document=False, pretty=True)

            print("Write image sets")
            with open(os.path.join(dataset_dirs["ImageSets"], "{}Step{}{}.txt".format(sets, group, ifa)), "w") as f:
                f.writelines(list(map(lambda x: str(x).zfill(12) + "\n", images.keys())))

            print("OK")
