import os
from pprint import pprint
from random import random

from easydict import EasyDict
from xmltodict import parse

cfg = EasyDict({
    "base_dir": os.path.join('data', 'VOCdevkit2007', 'VOC2007'),
    "nop_limit": 10000,
    "imdb_train": "voc_2007_trainval",
    "imdb_test": "voc_2007_test",
    "max_pic_train": 5,
    "max_pic_eval": 5,
    "label_names": ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    "nb_groups": 4,
    "factor": 0.
})

txt_base = os.path.join(cfg.base_dir, 'ImageSets', 'Main')
ann_base = os.path.join(cfg.base_dir, 'Annotations')

sets_name = ["tra", "val"]
sets_full_name = {
    "tra": "trainval",
    "val": "test"
}

nop_limit = cfg.nop_limit  # Abandon after many `do nothing`s

filename = {
    "tra": "{}.{}".format(cfg.imdb_train.split('_')[-1], "txt"),
    "val": "{}.{}".format(cfg.imdb_test.split('_')[-1], "txt")
}

max_pic = {
    "tra": cfg.max_pic_train,
    "val": cfg.max_pic_eval
}

total_file = {"tra": [], "val": []}

txt = {k: [x.strip() for x in open(os.path.join(txt_base, filename[k]))] for k in sets_name}
counter = {k: {x: 0 for x in cfg.label_names} for k in sets_name}


def sel(file_set, pic_max, cls_low, cls_hig, tol):
    counter_batch = {k: 0 for k in cfg.label_names}
    remain = set([k for k in cfg.label_names if cls_low <= cfg.label_names.index(k) < cls_hig])
    result = []
    do_nothing = 0
    while len(remain) or (len(result) < pic_max and do_nothing < nop_limit):
        do_nothing += 1
        line = file_set.pop(0)

        anno_path = os.path.join(ann_base, '{}.xml'.format(line))
        objs = parse(open(anno_path).read())['annotation']['object']
        objs = objs if isinstance(objs, list) else [objs]
        obj_cls = list(set([o['name'] for o in objs]))

        for name in obj_cls:
            if cfg.label_names.index(name) >= cls_hig:  # 1. Class range
                file_set.append(line)
                break
            if counter_batch[name] > pic_max:  # 2. Number of images
                file_set.append(line)
                break
            if cfg.label_names.index(name) < cls_low and random() > tol:  # 3, Leaky
                file_set.append(line)
                break
        else:
            result.append(line)
            for name in obj_cls:
                counter_batch[name] += 1
            remain.discard(name)
            do_nothing = 0
    return result, counter_batch, file_set


def write_to_file(fnm, data):
    with open(fnm, 'w') as f:
        f.writelines(["{}\n".format(x) for x in data])


for group in range(cfg.nb_groups):
    print('=' * 10, "Group", group, '=' * 10)
    sta = int(len(cfg.label_names) * 1. / cfg.nb_groups * group)
    fin = int(len(cfg.label_names) * 1. / cfg.nb_groups * (group + 1))

    for stage in sets_name:
        print(stage.capitalize(), end="...")
        files, counter_g, txt[stage] = sel(txt[stage], max_pic[stage], sta, fin, cfg.factor / (group + 1))
        print("OK")

        print(len(files), counter_g)

        total_file[stage] += files

        for k, v in counter_g.items():
            counter[stage][k] += v

        write_to_file(os.path.join(txt_base, '{}Step{}.txt'.format(sets_full_name[stage], group)), files)
        write_to_file(os.path.join(txt_base, '{}Step{}a.txt'.format(sets_full_name[stage], group)), total_file[stage])

print('=' * 10, "Summary", '=' * 10)
print("\t".join(["{}:{}".format(x, len(total_file[x])) for x in sets_name]))
pprint(counter)
