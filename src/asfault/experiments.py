import logging as l
import csv
import json
import random
import time

from asfault import config as c
from asfault.beamer import *
from asfault.evolver import *
from asfault.plotter import *
from asfault.tests import *
from asfault.repair_crossover import *


CSV_HEADER = [
    'TimeStamp',
    'FitnessFunction',
    'Boundary',
    'Aggression',
    'Generation',
    'Fitness',
    'OOB',
    'Timeouts',
    'Diversity',
    'MinTestID',
    'MaxTestID',
    'EvoTime',
    'TotalDiversity',
]


def plot_test(plot_file, test):
    title = 'Test: {}'.format(test.test_id)
    plotter = StandaloneTestPlotter(title, test.network.bounds)
    plotter.plot_test(test)
    save_plot(plot_file, dpi=c.pt.dpi_final)


def export_test_gen(plots_dir, tests_dir, test, render=False):
    plot_file = 'gen_{:04}.png'.format(test.test_id)
    plot_file = os.path.join(plots_dir, plot_file)
    test_file = 'test_{:04}.json'.format(test.test_id)
    test_file = os.path.join(tests_dir, test_file)

    if render:
        plot_test(plot_file, test)

    test_dict = RoadTest.to_dict(test)
    with open(test_file, 'w') as out:
        out.write(json.dumps(test_dict, sort_keys=True, indent=4))


def export_test_exec(plots_dir, execs_dir, test, render=False):
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if not os.path.exists(execs_dir):
        os.makedirs(execs_dir)

    plot_file = 'exe_{:04}.png'.format(test.test_id)
    plot_file = os.path.join(plots_dir, plot_file)
    test_file = 'test_{:04}.json'.format(test.test_id)
    test_file = os.path.join(execs_dir, test_file)

    if render:
        plot_test(plot_file, test)

    test_dict = RoadTest.to_dict(test)
    with open(test_file, 'w') as out:
        out.write(json.dumps(test_dict, sort_keys=True, indent=4))


def get_best_test(suite):
    max_score = -2
    max_test = None
    for test in suite:
        if test.score > max_score:
            max_score = test.score
            max_test = test
    return max_test


def get_worst_test(suite):
    min_score = sys.maxsize
    min_test = None
    for test in suite:
        if test.score < min_score:
            min_score = test.score
            min_test = test
    return min_test


def run_experiment(rng, evaluator, selector, estimator, factory, sort_pop, budget, time_limit=-1, random_exp=False, render=True, show=False):
    plots_dir = c.rg.get_plots_path()
    tests_dir = c.rg.get_tests_path()
    execs_dir = c.rg.get_execs_path()

    exported_tests_gen = set()
    exported_tests_exec = set()

    gen = TestSuiteGenerator(rng, evaluator, selector, estimator, factory,
                         sort_pop=sort_pop, cutoff=2 ** 64)

    if c.ev.attempt_repair:
        l.info("REPAIR: Enabled")
        gen.joiner = RepairJoin(rng, c.ev.bounds)
    else:
        l.info("REPAIR: Disabled")

    gen.random_exp = random_exp

    generation = 0
    evo_step = 0

    plotter = None
    if show:
        plotter = EvolutionPlotter()
        plotter.start()

    # data contains the current population of tests.
    for step, data in gen.evolve_suite(budget, time_limit=time_limit):
        evo_step += 1
        evo = list()

        if plotter:
            updated = plotter.update((step, data))
            if updated:
                plotter.pause()
                if render:
                    out_file = 'step_{:09}_{}.png'.format(evo_step, step)
                    out_file = os.path.join(plots_dir, out_file)
                    save_plot(out_file, dpi=c.pt.dpi_intermediate)

        if step == 'finish_generation' or step == 'looped':
            # Dump the population to log file
            l.warning("GENERATION %s: %s", "{:03d}".format(generation), ", ".join([str(test.test_id) for test in data]))

            for test in data:
                if test not in exported_tests_gen:
                    export_test_gen(plots_dir, tests_dir, test, render)
                    exported_tests_gen.add(test)

        if step == 'finish_evolution':
            final_path = c.rg.get_final_path()

            # Dump the population to log file
            l.warning("FINAL TEST SUITE: %s", ", ".join([str(test.test_id) for test in data]))

            for test in data:
                export_test_exec(final_path, final_path, test, render)

        if step == 'evaluated':
            generation += 1

            population, evaluation, total_evol, total_eval = data

            for test in population:
                if test.test_id not in exported_tests_exec:
                    export_test_exec(plots_dir, execs_dir, test, render)
                    exported_tests_exec.add(test.test_id)

            fitness = evaluation.score
            oob = evaluation.oob
            timeouts = evaluation.timeouts
            diversity = evaluation.coverage
            min_test = get_worst_test(population).test_id
            max_test = get_best_test(population).test_id

            evo.extend([generation, fitness, oob, timeouts,
                        diversity, min_test, max_test])
            evo.extend([total_evol, evaluation.total_coverage])

            yield evo


def experiment_out(rng, evaluator, selector, estimator, factory, sort_pop, budget, time_limit=-1, random_exp=False, render=False, show=False):
    out_file = c.rg.get_results_path()
    with open(out_file, 'w'):
        pass
    for evolution in run_experiment(rng, evaluator, selector, estimator, factory, sort_pop, budget,
                                    time_limit=time_limit,  random_exp=random_exp, render=render, show=show):
        out_file = c.rg.get_results_path()
        now_time = datetime.datetime.now()
        now_time = now_time.isoformat()
        prefix = [now_time, c.ev.evaluator, c.ev.bounds, c.ex.risk]
        evolution = prefix + evolution
        with open(out_file, 'a') as out:
            writer = csv.writer(out, delimiter=';',
                                quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(evolution)


def experiment(seed, budget, time_limit=-1, render=False, show=False, factory=None):
    rng = random.Random()
    rng.seed(seed)

    sort_pop = True

    level_dir = c.ex.get_level_dir()

    host = c.ex.host
    port = c.ex.port

    if factory is None:
        _factory = gen_beamng_runner_factory(level_dir, host, port, plot=show)
    else:
        _factory = factory

    random_exp = False

    if c.ev.evaluator == 'random':
        random_exp = True
        evaluator = RandomEvaluator(rng)
    elif c.ev.evaluator == 'uniqlanedist':
        evaluator = UniqueLaneDistanceEvaluator()
    else:
        evaluator = LaneDistanceEvaluator()

    if c.ev.selector == 'random':
        selector = RandomMateSelector(rng)
        sort_pop = False
    elif c.ev.selector == 'tournament':
        selector = TournamentSelector(rng, 2)
    else:
        print('Illegal value for selector: {}'.format(c.ex.selector))
        sys.exit(-1)

    if c.ev.estimator == 'random':
        estimator = RandomPathEstimator(rng)
    elif c.ev.estimator == 'length':
        estimator = LengthEstimator()
    else:
        print('Illegal value for estimator: {}'.format(c.ex.estimator))
        sys.exit(-1)

    experiment_out(rng, evaluator, selector, estimator, _factory,
                   sort_pop, budget, time_limit=time_limit,  random_exp=random_exp, render=render, show=show)
