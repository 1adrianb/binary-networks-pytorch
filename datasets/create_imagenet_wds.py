import sys
import os
import os.path
import random
import argparse

from torchvision import datasets

import webdataset as wds


parser = argparse.ArgumentParser("""Generate sharded dataset from original ImageNet data.""")
parser.add_argument("--splits", default="train,val", help="which splits to write")
parser.add_argument(
    "--filekey", action="store_true", help="use file as key (default: index)"
)
parser.add_argument("--maxsize", type=float, default=1e9)
parser.add_argument("--maxcount", type=float, default=10000)
parser.add_argument(
    "--shards", default="/home/dev/data_main/CORESETS/TinyImagenet_wds_more", help="directory where shards are written"
)
parser.add_argument(
    "--data",
    default="/home/dev/data_main/CORESETS/TinyImagenet/tiny-224",
    help="directory containing ImageNet data distribution suitable for torchvision.datasets",
)
args = parser.parse_args()


assert args.maxsize > 10000000
assert args.maxcount < 1000000


if not os.path.isdir(os.path.join(args.data, "train")):
    print(f"{args.data}: should be directory containing ImageNet", file=sys.stderr)
    print(f"suitable as argument for torchvision.datasets.ImageNet(...)", file=sys.stderr)
    sys.exit(1)


if not os.path.isdir(os.path.join(args.shards, ".")):
    print(f"{args.shards}: should be a writable destination directory for shards", file=sys.stderr)
    sys.exit(1)


splits = args.splits.split(",")


def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()


all_keys = set()


def write_dataset(imagenet, base="./shards", split="train"):

    # We're using the torchvision ImageNet dataset
    # to parse the metadata; however, we will read
    # the compressed images directly from disk (to
    # avoid having to reencode them)
    path = os.path.join(imagenet, split) 
    ds = datasets.ImageFolder(path)
    nimages = len(ds.imgs)
    print("# nimages", nimages)

    # We shuffle the indexes to make sure that we
    # don't get any large sequences of a single class
    # in the dataset.
    indexes = list(range(nimages))
    random.shuffle(indexes)

    # This is the output pattern under which we write shards.
    pattern = os.path.join(base, f"imagenet-{split}-%06d.tar")

    with wds.ShardWriter(pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount)) as sink:
        for i in indexes:

            # Internal information from the ImageNet dataset
            # instance: the file name and the numerical class.
            fname, cls = ds.imgs[i]
            assert cls == ds.targets[i]

            # Read the JPEG-compressed image file contents.
            image = readfile(fname)

            # Construct a uniqu keye from the filename.
            key = os.path.splitext(os.path.basename(fname))[0]

            # Useful check.
            assert key not in all_keys
            all_keys.add(key)

            # Construct a sample.
            xkey = key if args.filekey else "%07d" % i
            sample = {"__key__": xkey, "jpg": image, "cls": cls}

            # Write the sample to the sharded tar archives.
            sink.write(sample)


for split in splits:
    print("# split", split)
    write_dataset(args.data, base=args.shards, split=split)