#!/usr/bin/env python

import shutil
from glob import glob

for i, f in enumerate(glob('../input/train/*.*.jpg')):
    if 'dog' in f:
        shutil.move(f, '../input/train/dogs/{}.jpg'.format(i))
    elif 'cat' in f:
        shutil.move(f, '../input/train/cats/{}.jpg'.format(i))
