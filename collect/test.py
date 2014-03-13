#!/usr/bin/env python

import datetime
import time
import json
import dateutil.parser

x = datetime.datetime.now()

time.sleep(2)

y = datetime.datetime.now()

c = y-x

print c.days
print c.seconds

data = {}
data['id'] = y

def date_handler(obj):
    return obj.isoformat() if hasattr(obj, 'isoformat') else obj
 
tmp = json.dumps(data, default=date_handler)

print tmp

pd = json.loads(tmp)

for k in pd.keys():
	pd[k] = dateutil.parser.parse(pd[k])
