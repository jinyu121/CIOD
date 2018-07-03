import os
from collections import Counter
from pprint import pprint

from easydict import EasyDict
from xmltodict import parse

cfg = EasyDict({
    "base_dir": os.path.join('data', 'VOCdevkit2007', 'VOC2007'),
    "nop_limit": 10000,
    "imdb_tra": "trainval",
    "imdb_val": "test",
    "max_pic_tra": 20000,
    "max_pic_val": 20000,
    "label_names": [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ],
    "nb_groups": 4
})

sets_name = ["tra", "val"]

txt_base = os.path.join(cfg.base_dir, 'ImageSets', 'Main')
ann_base = os.path.join(cfg.base_dir, 'Annotations')


def prepare(filename):
    print("Prepare", filename)
    ids = [x.strip() for x in open(os.path.join(txt_base, filename + ".txt"))]
    mem = {}
    for line in ids:
        anno_path = os.path.join(ann_base, '{}.xml'.format(line))
        anno = parse(open(anno_path).read())['annotation']
        # Ignore the images who do not contain any objects
        objs = anno.get('object', [])
        if not objs:
            print("No object:", line)
            continue
        objs = objs if isinstance(objs, list) else [objs]
        obj_cls = [cfg.label_names.index(o['name']) for o in objs]
        mem[line] = set(obj_cls)
    print("Prepare", "OK", "Images", "{}/{}".format(len(mem), len(ids)))
    return mem


def sel(mem, pic_max, cls_low, cls_hig, nop_limit=1000):
    counter_batch = Counter()
    remain = set(list(range(cls_low, cls_hig)))
    result = []
    do_nothing = 0
    lines = list(mem.keys())

    while len(remain) or (len(result) < pic_max and do_nothing < nop_limit and len(lines)):
        do_nothing += 1
        line = lines.pop(0)
        objs = mem[line]

        for item in objs:
            if item >= cls_hig:  # 1. Class range
                lines.append(line)
                break
            if counter_batch[item] > pic_max:  # 2. Number of images
                lines.append(line)
                break
        else:
            result.append(line)
            counter_batch.update(list(objs))
            remain = remain - mem[line]
            del mem[line]
            do_nothing = 0
    return result, counter_batch, mem


def write_to_file(fnm, data):
    with open(fnm, 'w') as f:
        f.writelines(["{}\n".format(x) for x in data])


data = {stage: prepare(cfg["imdb_{}".format(stage)]) for stage in sets_name}
total_file = {stage: [] for stage in sets_name}

counter = {stage: Counter() for stage in sets_name}

for group in range(cfg.nb_groups):
    print('=' * 10, "Group", group, '=' * 10)
    sta = int(len(cfg.label_names) * 1. / cfg.nb_groups * group)
    fin = int(len(cfg.label_names) * 1. / cfg.nb_groups * (group + 1))

    for stage in sets_name:
        print(stage.capitalize())
        # sta = 0 if "val" == stage else sta
        files, counter_g, data[stage] = sel(data[stage], cfg["max_pic_{}".format(stage)], sta, fin, cfg.nop_limit)
        print(len(files), counter_g)

        total_file[stage] += files
        counter[stage].update(counter_g)
        write_to_file(os.path.join(txt_base, '{}Step{}.txt'.format(cfg["imdb_{}".format(stage)], group)), files)
        write_to_file(os.path.join(txt_base, '{}Step{}a.txt'.format(cfg["imdb_{}".format(stage)], group)),
                      total_file[stage])

print('=' * 10, "Summary", '=' * 10)
print("\t".join(["{}:{}".format(x, len(total_file[x])) for x in sets_name]))
pprint(counter)
