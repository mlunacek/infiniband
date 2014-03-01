#!/usr/bin/env python

from __future__ import print_function

import glob
import os

import parse
import config

files = glob.glob('../collect/test-*/results*')

map(parse.parse_file, files)

files = glob.glob(os.path.join(config.data_path,'*'))

map(print, files)

