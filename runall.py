import os
import shutil
from argparse import ArgumentParser

import torch
from tqdm import tqdm, trange

parser = ArgumentParser(description='Train a Fast R-CNN network')
parser.add_argument('-s', '--session', help='Session', type=int)
parser.add_argument('-a', '--all', action='store_true', help='All, or just One group')
parser.add_argument('-g', '--group', help='Group', type=int)
parser.add_argument('-t', '--train', action='store_true', help='Train')
parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate')
parser.add_argument('-ap', '--append', action='store_true', help='Data append')
parser.add_argument('-la', '--log_all', action='store_true', help='Log all')
args = parser.parse_args()

session = args.session
num_gropus = args.group

batch_size = 4 * torch.cuda.device_count()

index_folder = "data/VOCdevkit2007/VOC2007/ImageSets/Main"
result_dir = "results/res101/pascal_voc"
cache_by_rcnn = "data/cache"


def delf(p):
    try:
        os.remove(p)
    except:
        pass


if args.all:
    sta, fin = 0, num_gropus + 1
else:
    sta, fin = num_gropus, num_gropus + 1
print("Will train {}".format(range(sta, fin)))

if args.train:
    delf("log_train_s{}.log".format(session))
if args.evaluate:
    delf("log_test_s{}.log".format(session))

for group in trange(sta, fin):
    # clean train cache
    delf(os.path.join(cache_by_rcnn, "voc_2007_trainval_gt_roidb.pkl"))
    delf(os.path.join(cache_by_rcnn, "voc_2007_test_gt_roidb.pkl"))
    delf(os.path.join(index_folder, "test.txt_annots.pkl"))

    # copy train file
    src = os.path.join(index_folder, "trainvalStep{}{}.txt".format(group, "a" if args.append else ""))
    dst = os.path.join(index_folder, "trainval.txt")
    delf(dst)
    shutil.copyfile(src, dst)

    # copy test file, clean test cache
    src = os.path.join(index_folder, "testStep{}a.txt".format(group))
    dst = os.path.join(index_folder, "test.txt")
    delf(dst)
    shutil.copyfile(src, dst)

    # train
    if args.train:
        tqdm.write("Group {} Train".format(group))
        cmd = "python trainval_net.py \
                    --cuda --mGPUs \
                    --dataset pascal_voc \
                    --net res101 \
                    --bs {} \
                    --s {} \
                    --nw 32 \
                    --lr 0.001 \
                    --epochs 10 \
                    --lr_decay_step 8 \
                    --group {} \
                    --save_dir results  >> log_train_s{}.log {}".format(batch_size, session, group, session,
                                                                        "2>&1" if args.log_all else "")
        os.system(cmd)

    # test
    if args.evaluate:
        tqdm.write("Group {} Test".format(group))
        cmd = "python test_net.py \
                    --dataset pascal_voc \
                    --net res101 \
                    --checksession {} \
                    --group {} \
                    --load_dir results \
                    --cuda >> log_test_s{}.log {}".format(session, group, session, "2>&1" if args.log_all else "")
        os.system(cmd)

    tqdm.write("Group {} Done".format(group))
