import csv
import dateutil.parser
import glob
import logging as l
import json
import random
import shutil
from pathlib import Path
from time import time
import os.path
import itertools

from collections import defaultdict
from matplotlib import pyplot as plt
import click
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import scipy.stats

from asfault import config, experiments
from asfault.beamer import *
from asfault.network import *
from asfault.evolver import *
from asfault.graphing import *
from asfault.plotter import *

BEAMNG_FILES = 'beamng_templates'

RESULTS_FILE = 'results.json'


DEFAULT_LOG = 'asfault.log'
DEFAULT_ENV = os.path.join(str(Path.home()), '.asfaultenv')


def log_exception(extype, value, trace):
    l.exception('Uncaught exception:', exc_info=(extype, value, trace))


def setup_logging(log_file):
    file_handler = l.FileHandler(log_file, 'a', 'utf-8')
    term_handler = l.StreamHandler()
    l.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                  level=l.DEBUG, handlers=[term_handler, file_handler])
    sys.excepthook = log_exception
    l.info('Started the logging framework writing to file: %s', log_file)


def milliseconds():
    return round(time() * 1000)


def read_environment(env_dir):
    l.info('Starting with environment from: %s', env_dir)
    config.load_configuration(env_dir)


def ensure_environment(env_dir):
    if not os.path.exists(env_dir):
        l.info('Initialising empty environment: %s', env_dir)
        config.init_configuration(env_dir)
    read_environment(env_dir)


@click.group()
@click.option('--log', type=click.Path(dir_okay=False), default=DEFAULT_LOG)
def cli(log):
    setup_logging(log)
    generate_factories()


@cli.group()
@click.option('--env', type=click.Path(file_okay=False), default=DEFAULT_ENV)
@click.option('--flush-output', is_flag=True)
def evolve(env, flush_output):
    ensure_environment(env)
    if flush_output:
        output_dir = config.rg.get_output_path()
        shutil.rmtree(output_dir)
        config.rg.ensure_directories()


@evolve.command()
@click.option('--seed', default=milliseconds())
@click.option('--budget', default=100)
@click.option('--render', is_flag=True)
@click.option('--show', is_flag=True)
def bng(seed, budget, render, show):
    l.info('Starting experiment with seed: %s', seed)
    l.info('Starting experiment with budget: %s', budget)
    experiments.experiment(seed, budget, render, show)


@evolve.command()
@click.option('--seed', default=milliseconds())
@click.option('--generations', default=10)
@click.option('--render', is_flag=True)
@click.option('--show', is_flag=True)
@click.option('--uniq', is_flag=True)
@click.argument('ctrl')
def ext(seed, generations, render, uniq, ctrl):
    l.info('Starting external evolve with seed: {}'.format(seed))
    l.info('AI Controller call given is: %s', ctrl)
    plots_dir = config.rg.get_plots_path()
    tests_dir = config.rg.get_tests_path()

    level_dir = config.ex.get_level_dir()
    port = config.ex.port

    config.ev.join_probability = 1.1
    config.ev.lane_width = 4.3
    if uniq:
        config.ev.evaluator = 'uniqlanedist'
    else:
        config.ev.evaluator = 'lanedist'
    config.ev.try_all_ops = False
    config.ex.max_speed = 'false'
    config.ex.ai_controlled = 'false'
    config.ex.direction_agnostic_boundary = True

    rng = random.Random()
    rng.seed(seed)

    factory = gen_beamng_runner_factory(
        level_dir, config.ex.host, port, False, ctrl=ctrl)
    evaluator = LaneDistanceEvaluator()
    selector = TournamentSelector(rng, 2)
    estimator = TurnAndLengthEstimator()

    experiments.experiment_out(rng, evaluator, selector, estimator, factory,
                               True, generations, render=render, show=show)


@evolve.command()
@click.option('--seed', default=milliseconds())
@click.option('--generations', default=10)
@click.option('--show', default=False)
@click.option('--render', is_flag=True)
def mock(seed, generations, show, render):
    plots_dir = config.rg.get_plots_path()
    tests_dir = config.rg.get_tests_path()

    if show or render:
        plotter = EvolutionPlotter()
        if show:
            plotter.start()
    else:
        plotter = None

    rng = random.Random()
    rng.seed(seed)

    factory = gen_mock_runner_factory(rng)
    evaluator = StructureEvaluator()
    selector = TournamentSelector(rng, 2)
    estimator = LengthEstimator()

    gen = TestSuiteGenerator(rng, evaluator, selector, estimator, factory)

    step = 0
    for state in gen.evolve_suite(generations):
        if plotter:
            updated = plotter.update(state)
            if updated:
                if show:
                    plotter.pause()
                if render:
                    out_file = '{:08}.png'.format(step)
                    out_file = os.path.join(plots_dir, out_file)
                    save_plot(out_file, dpi=c.pt.dpi_intermediate)
                    step += 1

    suite = gen.population

    for test in suite:
        test_file = os.path.join(tests_dir, '{0:08}.json'.format(test.test_id))
        plot_file = os.path.join(plots_dir,
                                 'final_{0:08}.png'.format(test.test_id))

        plotter = StandaloneTestPlotter('Test: {}'.format(test.test_id),
                                        test.network.bounds)
        plotter.plot_test(test)
        save_plot(plot_file, dpi=c.pt.dpi_final)

        test_dict = RoadTest.to_dict(test)
        with open(test_file, 'w') as out:
            out.write(json.dumps(test_dict, sort_keys=True, indent=4))
        clear_plot()

    for test in suite:
        continue
        map_file = os.path.join(
            plots_dir, 'map_{0:08}.png'.format(test.test_id))
        generate_road_mask(test.network, map_file,
                           buffer=4 * config.ev.lane_width)
        noise_file = os.path.join(
            plots_dir, 'noise_{0:08}.png'.format(test.test_id))
        generate_noise_road_map(random.Random(), 2048,
                                2048, 1024, 512, map_file, noise_file)

    out_dir = config.rg.get_output_path()
    out_file = os.path.join(out_dir, 'props.json')
    props = {'seed': seed}
    with open(out_file, 'w') as out:
        out.write(json.dumps(props, sort_keys=True, indent=4))


@evolve.command()
@click.argument('test-file', nargs=1)
def run_test(test_file):
    l.info('Starting BeamNG.research to run test: %s', test_file)
    with open(test_file, 'r') as infile:
        test_dict = json.loads(infile.read())
    test = RoadTest.from_dict(test_dict)
    out_dir = config.ex.get_level_dir()
    host = config.ex.host
    port = config.ex.port
    runner = TestRunner(test, out_dir, host, port, plot=True)
    execution = runner.run()


def process_oob_segs(oob_segs):
    summary = defaultdict(int)
    for seg_key, count in oob_segs.items():
        parts = seg_key.split('_')
        roadtype = parts[0]
        if roadtype == 'straight':
            roadtype = 'Straight'
            length = float(parts[1])
            if length < 100:
                length = 'Short'
            elif length < 200:
                length = 'Medium'
            else:
                length = 'Long'
            key = '{}\n{}'.format(roadtype, length)
            summary[key] += count

        if roadtype == 'l' or roadtype == 'r':
            if roadtype == 'l':
                roadtype = 'Left'
            else:
                roadtype = 'Right'

            angle = abs(float(parts[2]))
            pivot = float(parts[3])

            if angle < 45:
                angle = 'Gentle'
            else:
                angle = 'Sharp'

            if pivot <= 25:
                pivot = 'Narrow'
            else:
                pivot = 'Wide'

            key = '{}\n{}, {}'.format(roadtype, angle, pivot)
            summary[key] += count

    return summary


@cli.command()
@click.argument('exp-dir', type=click.Path(file_okay=False))
def process_results(exp_dir):
    l.info('Processing results in: %s', exp_dir)
    final_results = None
    config.load_configuration(exp_dir)
    config.rg.ensure_directories()
    results_file = config.rg.get_results_path()
    if os.path.exists(results_file):
        data = pd.read_csv(
            results_file, sep=';', quoting=csv.QUOTE_NONNUMERIC, names=experiments.CSV_HEADER)

        graph_oobs_over_gens(data, config.rg.get_oobs_gens_path())
        props = get_exp_properties(data)
        props_file = 'props.json'
        props_file = os.path.join(exp_dir, props_file)
        with open(props_file, 'w') as out_file:
            out_file.write(json.dumps(props, indent=4, sort_keys=True))

    execs_path = config.rg.get_execs_path()
    if os.path.exists(execs_path):
        oob_segs = defaultdict(int)
        oob_speeds = []
        for root, _, files in os.walk(execs_path, topdown=True):
            for fil in files:
                fil = os.path.join(root, fil)
                with open(fil, 'r') as in_file:
                    exec = json.loads(in_file.read())
                    exec = exec['execution']
                    if 'oob_speeds' in exec:
                        oob_speeds.extend(exec['oob_speeds'])
                    if 'seg_oob_count' in exec:
                        seg_oob_count = exec['seg_oob_count']
                        for key, val in seg_oob_count.items():
                            oob_segs[key] += val

        oob_segs = process_oob_segs(oob_segs)
        graph_oob_segs(oob_segs, config.rg.get_oob_segs_path())


if __name__ == '__main__':
    cli()
