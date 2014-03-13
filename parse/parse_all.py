#!/usr/bin/env python

from __future__ import print_function

import glob
import os
from functools import partial

import parse
import parse_ports_cables as ppc
import config

files = glob.glob('../collect/test-*/results*')
map(parse.parse_file, files)

# Parse the cables and ports
files = glob.glob(os.path.join(config.data_path,'results*.csv'))
map(ppc.parse_ports, files)
map(ppc.parse_cables, files)

# switches = partial(ppc.parse_cables, switches=True)
# map(switches, files)

files = glob.glob(os.path.join(config.data_path,'*.csv'))
map(print, files)


