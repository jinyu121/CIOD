import copy
import json
import os
from collections import Counter

from easydict import EasyDict

cfg = EasyDict({
    "base_dir": os.path.join('data', 'coco', 'annotations'),
    "nop_limit": 10000,
    "imdb_train": "instances_train2017",
    "imdb_test": "instances_val2017",
    "label_names": [],
    "nb_groups": 4
})

sets_name = ["tra", "val"]
sets_full_name = {"tra": "trainval", "val": "test"}

filename = {"tra": "{}.{}".format(cfg.imdb_train, "json"), "val": "{}.{}".format(cfg.imdb_test, "json")}
total_json = {x: json.load(open(os.path.join(cfg.base_dir, filename[x]))) for x in sets_name}
result_json = {x: {} for x in sets_name}

cfg.label_names = sorted([x['id'] for x in total_json['val']['categories']])
counters = {x: Counter() for x in sets_name}


def sel(data, cls_range):
    print("Before: Image {}, Object {}".format(len(data['images']), len(data['annotations'])))

    anno_pool = data['annotations']
    image_pool = {x['id']: x for x in data['images']}

    # Clean images who contain objects that not in range
    for anno in anno_pool:
        if anno['category_id'] not in cls_range:
            image_id = anno['image_id']
            if image_id in image_pool:
                del image_pool[image_id]

    # Double check: Delete objects who do not have father
    anno_pool = [anno for anno in anno_pool if anno['image_id'] in image_pool]
    reverse_pool = set(anno['image_id'] for anno in anno_pool)

    # Triple check if image have annotation
    image_pool = {k: v for k, v in image_pool.items() if k in reverse_pool}

    data['images'] = [v for k, v in image_pool.items()]
    data['annotations'] = anno_pool
    data['categories'] = [x for x in data['categories'] if x['id'] in cls_range]

    print("After: Image {}, Object {}".format(len(data['images']), len(data['annotations'])))
    d = [x['category_id'] for x in data['annotations']]
    counter = Counter(d)

    return data, counter


for group in range(cfg.nb_groups):
    print('=' * 10, "Group", group, '=' * 10)
    sta = int(len(cfg.label_names) * 1. / cfg.nb_groups * group)
    fin = int(len(cfg.label_names) * 1. / cfg.nb_groups * (group + 1))
    label_range = cfg.label_names[sta:fin]
    print(len(label_range), label_range)

    for stage in sets_name:
        print(stage.capitalize(), "...")
        data = copy.deepcopy(total_json[stage])
        data, cnt = sel(data, label_range)
        # ==========
        print(cnt)
        counters[stage] += cnt
        # ==========
        json.dump(data,
                  open(os.path.join(cfg.base_dir, '{}Step{}.json'.format(sets_full_name[stage], group)), "w"))
        # ==========
        if 0 == group:
            result_json[stage] = data
        else:
            for k in result_json[stage]:
                if isinstance(result_json[stage][k], list):
                    result_json[stage][k] += data[k]
        json.dump(result_json[stage],
                  open(os.path.join(cfg.base_dir, '{}Step{}a.json'.format(sets_full_name[stage], group)), "w"))

print(counters)
