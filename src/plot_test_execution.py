import glob
import os.path
import json
import math
import csv
import argparse
import tempfile

from shapely.geometry import Point, Polygon

from asfault.tests import CarState, RoadTest
from asfault.app import ensure_environment, DEFAULT_ENV
from asfault.tests import RoadTest
from asfault.plotter import StandaloneTestPlotter, save_plot
from asfault import config as c

ensure_environment(DEFAULT_ENV)

# inputJSON="/Volumes/ALESSIO/CompressExps/SingleExperiments/2018-07-03T12-35-59/003/003_lanedist_1000_0125/output/execs/test_2855.json"
# inputJSON="/Volumes/ALESSIO/CompressExps/SingleExperiments/2018-06-29T10-01-25/002/002_lanedist_1000_0075/output/execs/test_2855.json"
# inputJSON="/Users/gambi/Dropbox/MarcMuller/Exps/Multi/2018-09-06T08-02-03/000/000_lanedist_0500_0075/output/final/test_0087.json"
# inputJSON="/Users/gambi/Dropbox/MarcMuller/Exps/Multi/2018-09-06T08-02-03/000/000_lanedist_0500_0075/output/execs/test_0659.json"
#inputJSON="/Volumes/ALESSIO/CompressExps/SingleExperiments/2018-06-30T19-57-59/002/002_lanedist_1000_0075/output/execs/test_2917.json"
inputJSON ="/Users/gambi/AsFault-Implementation/src/unit_tests/test_2855.json"

with open(inputJSON) as handle:
    dictdump = json.loads(handle.read())

    # Parse the JSON to RoadTest
test = RoadTest.from_dict(dictdump)

print(test.test_id)
print(test.network.bounds)
# test_file = os.path.join(tests_dir, '{0:08}.json'.format(test.test_id))
# plot_file = os.path.join(plots_dir, 'final_{0:08}.png'.format(test.test_id))

bounds = test.network.bounds

coords = bounds.exterior.coords[:]
for idx, c in enumerate(coords):
    print("idx", idx, "c", c )
    coords[idx] = (c[0]*2.0, c[1]*2.0)

large_bounds = Polygon(coords);
plotter = StandaloneTestPlotter('Test: {}'.format(test.test_id), large_bounds)
# plotter.plot_test(test)
states = dictdump["execution"]["states"]
trace = []
for idx, state_dict in enumerate(states):
    # Parse input
    trace.append(CarState.from_dict(test, state_dict))

plotter.plot_test(test)
plotter.plot_car_trace(trace)
DPI_FINAL = 300
save_plot("test.png", dpi=600)
