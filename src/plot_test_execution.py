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


def plot_it(inputJSON, to, the_bounds=None):
    print("Plotting", inputJSON, "to", to)
    with open(inputJSON) as handle:
        dictdump = json.loads(handle.read())

        # Parse the JSON to RoadTest
    test = RoadTest.from_dict(dictdump)

    # print(test.test_id)
    # print(test.network.bounds)
    # test_file = os.path.join(tests_dir, '{0:08}.json'.format(test.test_id))
    # plot_file = os.path.join(plots_dir, 'final_{0:08}.png'.format(test.test_id))

    if the_bounds is None:
        bounds = test.network.bounds

        coords = bounds.exterior.coords[:]
        for idx, c in enumerate(coords):
            print("idx", idx, "c", c )
            coords[idx] = (c[0]*2.0, c[1]*2.0)
        large_bounds = Polygon(coords);
    else:
        large_bounds=the_bounds


    plotter = StandaloneTestPlotter('Test: {}'.format(test.test_id), large_bounds)
    # plotter.plot_test(test)
    states = dictdump["execution"]["states"]
    trace = []
    for idx, state_dict in enumerate(states):
        # Parse input
        trace.append(CarState.from_dict(test, state_dict))

    plotter.plot_test(test)
    plotter.plot_car_trace(trace)
    save_plot(to, dpi=100)

x_min = 300
x_max = 400

y_min = 400
y_max = 500

p = Polygon([[x_max, y_max], [x_min, y_max], [x_min, y_min], [x_max, y_min]])
print(p)

plot_it("/tmp/images/asfault_single/2/.asfaultenv/output/execs/test_0184.json", "test.png", the_bounds=p)


# counter = 1
# for inputJSON in glob.iglob('/tmp/images/**/.asfaultenv/**/execs/*.json', recursive=True):
#     print("Found", inputJSON)
#     plot_it(inputJSON, "test" + str(counter)+ ".png")
#     counter += 1


