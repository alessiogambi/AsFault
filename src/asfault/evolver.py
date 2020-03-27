import datetime
import logging as l
import random
import sys

from time import time, sleep

from shapely.geometry import box

from asfault import crossovers
from asfault import mutations
from asfault.config import *
from asfault.beamer import TestRunner, RESULT_SUCCESS, REASON_OFF_TRACK, REASON_TIMED_OUT, REASON_GOAL_REACHED, generate_test_prefab
from asfault.generator import RoadGenerator, generate_networks
from asfault.tests import *
from asfault.plotter import *
from asfault.network import SEG_FACTORIES

UNIQUE_THRESHOLD = 0.1

def plot_test(plot_file, test):
    title = 'Test: {}'.format(test.test_id)
    plotter = StandaloneTestPlotter(title, test.network.bounds)
    plotter.plot_test(test)
    save_plot(plot_file, dpi=c.pt.dpi_final)

def plot_network(plot_file, network):
    title = 'debug'
    plotter = StandaloneTestPlotter(title, network.bounds)
    plotter.plot_network(network)
    save_plot(plot_file, dpi=c.pt.dpi_final)

def get_lines_angle(linea, lineb):
    acoords = linea.coords
    bcoords = lineb.coords
    adir = (acoords[-1][0] - acoords[0][0], acoords[-1][1], acoords[0][1])
    bdir = (bcoords[-1][0] - bcoords[0][0], bcoords[-1][1], bcoords[0][1])
    xdiff = adir[0] - bdir[0]
    ydiff = adir[1] - bdir[1]
    angle = math.degrees(math.atan2(ydiff, xdiff))
    angle = math.fabs(angle)
    return angle


def score_path_polyline(path_line):
    coords = path_line.coords
    if len(coords) < 3:
        return 0.01

    sum = 0.01
    last_coord = coords[0]
    for coord, next_coord in zip(coords[1:], coords[2:]):
        beg = LineString([last_coord, coord])
        end = LineString([coord, next_coord])
        angle = get_lines_angle(beg, end)
        length = beg.length + end.length
        sum += angle
        sum += length

    sum = float(sum)
    sum /= 10000.0

    return sum


class PathEstimator:
    def score_path(self, polyline):
        raise NotImplementedError()


class RandomPathEstimator:
    def __init__(self, rng):
        self.rng = rng

    def score_path(self, path, polyline):
        return self.rng.uniform(0, 10)


class TurnAndLengthEstimator:
    def score_path(self, path, polyline):
        return score_path_polyline(polyline)

class LengthEstimator:
    def score_path(self, path, polyline):
        return len(path)


class TestSuiteEvaluation:
    RESULT_TIMEOUT = 'timeout'
    RESULT_SUCCESS = 'success'

    def __init__(self, result, score, oob, timeouts, coverage, total_coverage, duration):
        self.result = result
        self.score = score
        self.oob = oob
        self.timeouts = timeouts
        self.coverage = coverage
        self.total_coverage = total_coverage
        self.duration = duration


class SuiteEvaluator:
    def evaluate_suite(self, suite):
        eval_beg = datetime.datetime.now()
        min = sys.maxsize
        max = -1
        sum = 0

        oob = 0
        timeouts = 0

        for idx, test in enumerate(suite):
            l.info('Evaluating test: {}/{}'.format(idx + 1, len(suite)))
            score, reason = self.evaluate_test(test, suite)
            test.score = score
            test.reason = reason

            if score < min:
                min = score
            if score > max:
                max = score
            sum += score

            oob += test.execution.oobs

            if test.reason == REASON_TIMED_OUT:
                timeouts += 1

        eval_end = datetime.datetime.now()
        eval_dur = (eval_end - eval_beg).seconds

        coverage = RoadTest.get_suite_coverage(suite, 2)
        total_coverage = RoadTest.get_suite_total_coverage(suite, 2)

        ret = TestSuiteEvaluation(TestSuiteEvaluation.RESULT_SUCCESS, sum, oob, timeouts, coverage, total_coverage, eval_dur)
        return ret

    def evaluate_test(self, test, suite):
        raise NotImplementedError()


class StructureEvaluator(SuiteEvaluator):

    def evaluate_test(self, test, suite):
        l.info('Evaluating test: %s', str(test))
        path_line = test.get_path_polyline()
        return score_path_polyline(path_line) * len(test.network.nodes), RESULT_SUCCESS


class BeamNGEvaluator(SuiteEvaluator):
    def __init__(self,
                 level_dir=ExecutionConfig.BEAMNG_DIR,
                 host=ExecutionConfig.HOST,
                 port=ExecutionConfig.PORT,
                 plot=False):
        self.level_dir = level_dir
        self.host = host
        self.port = port
        self.plot = plot

    def evaluate_test(self, test, suite):
        if test.score:
            return test.score

        runner = TestRunner(test, self.level_dir, self.host, self.port, self.plot)
        try:
            execution = runner.run()
        finally:
            runner.close()
        if execution.average_distance:
            return execution.maximum_distance, execution.oobs
        else:
            return -1, -1

class MockRunner:
    def __init__(self, rng, test):
        self.rng = rng
        self.test = test

    def run(self):
        now = datetime.datetime.now()
        return TestExecution(self.test, self.rng.random(), REASON_GOAL_REACHED, [], self.rng.randint(0, 5), now, now, minimum_distance=0, average_distance=0.5, maximum_distance=1.0)

    def close(self):
        pass

def gen_mock_runner_factory(rng):
    def factory(test):
        runner = MockRunner(rng, test)
        return runner
    return factory

class LaneDistanceEvaluator(SuiteEvaluator):
    def evaluate_test(self, test, suite):
        if test.execution.result == RESULT_SUCCESS:
            fitness = test.execution.maximum_distance / c.ev.lane_width
            fitness = min(fitness, 1.0)
        else:
            fitness = -1

        return fitness, test.execution.reason

class SegCountEvaluator(SuiteEvaluator):
    def evaluate_test(self, test, suite):
        return len(test.network.parentage.nodes()), RESULT_SUCCESS

class UniquenessEvaluator(SuiteEvaluator):

    def evaluate_test(self, test, suite):
        ret = 0
        for other in suite:
            if other == test:
                continue

            testdist = test.distance(other)
            ret += testdist
        ret /= (len(suite) - 1)
        return ret, RESULT_SUCCESS

class UniqueLaneDistanceEvaluator(SuiteEvaluator):
    def __init__(self):
        self.lanedist = LaneDistanceEvaluator()
        self.uniq = UniquenessEvaluator()

    def evaluate_test(self, test, suite):
        score_lanedist, reason = self.lanedist.evaluate_test(test, suite)
        score_uniq, reason = self.uniq.evaluate_test(test, suite)
        return score_uniq * score_lanedist, reason

class RandomEvaluator(SuiteEvaluator):
    def __init__(self, rng):
        self.rng = rng

    def evaluate_test(self, test, suite):
        return self.rng.random(), test.execution.reason

class MateSelector:
    def select(self, suite, ignore=set()):
        raise NotImplementedError()


class RandomMateSelector(MateSelector):
    def __init__(self, rng):
        self.rng = rng

    def select(self, suite, ignore=set()):
        options = [resident for resident in suite if resident not in ignore]
        return self.rng.sample(options, 1)[0]


class TournamentSelector(MateSelector):
    def __init__(self, rng, tourney_size):
        self.rng = rng
        self.tourney_size = tourney_size

    def select(self, suite, ignore=set()):
        suite = [test for test in suite if test not in ignore]
        best = None
        for i in range(self.tourney_size):
            if suite:
                idx = self.rng.randint(0, len(suite) - 1)
                resident = suite.pop(idx)
                l.info('Resident: %s', str(resident))
                assert resident.score
                if not best or resident.score > best.score:
                    best = resident
        return best

class TestSuiteGenerator:

    def __init__(self,
                 rng=random.Random(),
                 evaluator=StructureEvaluator(),
                 selector=TournamentSelector(random.Random(), 3),
                 estimator=TurnAndLengthEstimator(),
                 runner_factory=None,
                 sort_pop=True,
                 plotter=None,
                 **evopts):
        self.rng = rng
        self.evaluator = evaluator
        self.selector = selector
        self.estimator = estimator
        self.runner_factory = runner_factory
        self.sort_pop = sort_pop
        self.plotter = plotter
        self.evopts = evopts

        size = c.ev.bounds
        self.bounds = box(-size, -size, size, size)
        self.max_pop = c.ev.pop_size

        self.population = []

        self.test_id = 1
        self.step = 0

        self.random_exp = False

        self.merger = crossovers.PartialMerge(self.rng)
        self.joiner = crossovers.Join(self.rng)

        self.mutators = {}
        self.init_mutators()
        self.sg_idx = 0

    def init_crossovers(self):
        pass
        # for m_count in range(1, 5):
        #     for d_count in range(1, 5):
        #         key = 'partial_crossover_{}x{}'.format(m_count, d_count)
        #         self.crossovers[key] = crossovers.PartialMerge(self.rng, m_count=m_count, d_count=d_count)

        #self.crossovers['partial_crossover_random'] = crossovers.PartialMerge(self.rng)
        #self.crossovers['join'] = crossovers.Join(self.rng)

    def init_mutators(self):
        for key, factory in SEG_FACTORIES.items():
            name = 'repl_' + key
            self.mutators[name] = mutations.SegmentReplacingMutator(self.rng, name, key, factory)

    def next_seed(self):
        return self.rng.randint(0, 2 ** 32 - 1)

    def next_test_id(self):
        ret = self.test_id
        self.test_id += 1
        return ret

    def run_test(self, test):
        while True:
            runner = self.runner_factory(test)
            try:
                execution = runner.run()
                return execution
            except Exception as e:
                l.error('Error running test %s', test)
                l.exception(e)
                sleep(30.0)
            finally:
                runner.close()

    def run_suite(self):
        for test in self.population:
            if not test.execution:
                execution = self.run_test(test)
                test.execution = execution

    def spawn_test(self, bounds, seed):
        network = generate_networks(bounds, [seed])[0]
        test = self.test_from_network(network)
        return test

    def test_from_network(self, network):
        start, goal, path = self.determine_start_goal_path(network)
        l.debug('Got start, goal, and path for network.')
        test = RoadTest(self.next_test_id(), network, start, goal)
        if path:
            l.debug('Setting path of new test: %s', test.test_id)
            test.set_path(path)
            l.debug('Set path of offspring.')
        return test

    def attempt_crosssover(self, mom, dad, crossover):
        l.debug('Attempting to cross over: %s (%s) %s', mom, type(crossover), dad)
        epsilon = 0.01
        while True:
            children, aux = crossover.apply(mom, dad)
            if children:
                consistent = []
                for child in children:
                    try:
                        if child.complete_is_consistent():
                            test = self.test_from_network(child)
                            consistent.append(child)
                        else:
                            break
                    except Exception as e:
                        l.error('Exception while creating test from child: ')
                        #l.exception(e)
                        break

                if consistent:
                    return consistent, aux

            failed = self.rng.random()
            if failed < epsilon:
                break

            epsilon *= 1.05

        return None, aux

    def crossover(self, mom, dad):
        choice = self.rng.random()
        choices = []

        # l.debug('Picking crossover %s < %s ?', choice, c.ev.join_probability)
        if choice < c.ev.join_probability:
            choices = [self.joiner]
            l.info('Crossover using for %s', choice)
        else:
            if c.ev.try_all_ops:
                choices.append(self.merger)
                l.info('Crossover using for %s', choice)
            else:
                l.info('Skip Crossover')

        for choice in choices:
            children, aux = choice.try_all(mom, dad, self)
            l.info('Finished trying all crossovers')

            #children, aux = self.attempt_crosssover(mom, dad, choice)
            if not aux:
                aux = {}
            if children:
                aux['type'] = choice
                l.debug('Cross over was applicable.')
                tests = []
                for child in children:
                    if child.complete_is_consistent():
                        test = self.test_from_network(child)
                        tests.append(test)
                return tests, aux

        l.debug('Cross over between %s x %s considered impossible.', mom, dad)
        return None, {}

    def roll_mutation(self, resident):
        return self.rng.random() <= c.ev.mut_chance

    def roll_introduction(self):
        return self.rng.random() <= c.ev.intro_chance

    def attempt_mutation(self, resident, mutator):
        epsilon = 0.25
        while True:
            try:
                mutated, aux = mutator.apply(resident)
                if mutated and mutated.complete_is_consistent():
                    test = self.test_from_network(mutated)
                    return mutated, aux
            except Exception as e:
                l.error('Exception while creating test from child: ')
                l.exception(e)
            failed = self.rng.random()
            if failed < epsilon:
                break

            epsilon *= 1.1

        return None, {}

    def mutate(self, resident):
        mutators = [*self.mutators.values()]
        self.rng.shuffle(mutators)
        while mutators:
            mutator = mutators.pop()
            mutated, aux = self.attempt_mutation(resident, mutator)
            if not aux:
                aux = {}

            aux['type'] = mutator
            if mutated:
                test = self.test_from_network(mutated)
                l.info('Successfully applied mutation: %s', str(type(mutator)))
                return test, aux

        return None, {}

    def determine_start_goal_path(self, network):
        best_start, best_goal = None, None
        best_path = None
        best_score = -1

        epsilon = 0.1
        candidates = list(network.get_start_goal_candidates())
        self.rng.shuffle(candidates)
        candidate_idx = 0
        sg_file = 'sg_{:08}.png'.format(self.sg_idx)
        sg_file = os.path.join(c.rg.get_plots_path(), sg_file)
        sg_json = 'sg_{:08}.json'.format(self.sg_idx)
        sg_json = os.path.join(c.rg.get_plots_path(), sg_json)
        #with open(sg_json, 'w') as out_file:
            #out_file.write(json.dumps(NetworkLayout.to_dict(network), indent=4, sort_keys=True))
        self.sg_idx += 1
        #plot_network(sg_file, network)
        if candidates:
            for start, goal in candidates:
                #l.info(sg_file)
                l.info('Checking candidate: (%s, %s), %s/%s', start, goal, candidate_idx, len(candidates))
                candidate_idx += 1
                paths = network.all_paths(start, goal)
                #paths = network.all_shortest_paths(start, goal)
                start_coord, goal_coord = get_start_goal_coords(network, start, goal)
                i = 0
                done = 0.05
                for path in paths:
                    l.info('Path has length: %s', len(path))
                    try:
                        polyline = get_path_polyline(network, start_coord, goal_coord, path)
                    except:
                        break
                    l.info('Got polyline.')
                    score = self.estimator.score_path(path, polyline)
                    l.info('Got score estimation: %s', score)
                    if score > best_score:
                        best_start = start
                        best_goal = goal
                        best_path = path
                        best_score = score
                    i += 1

                    done = self.rng.random()
                    if done < epsilon:
                        break

                    epsilon *= 1.25
                if done < epsilon:
                    break

            best_start, best_goal = get_start_goal_coords(network, best_start,
                                                          best_goal)

            return best_start, best_goal, best_path

        return None, None, None

    def determine_path(self, test):
        start_node = test.network.get_nodes_at(test.start)
        goal_node = test.network.get_nodes_at(test.goal)
        assert len(start_node) > 0
        assert len(goal_node) > 0
        start_node = start_node.pop()
        goal_node = goal_node.pop()
        return test.network.shortest_path(start_node, goal_node)

    def generate_single_test(self):
        network = generate_networks(self.bounds, [self.next_seed()])[0]
        test = self.test_from_network(network)
        return test

    def generate_tests(self, amount):
        ret = []
        todo = []
        generators = {}
        for i in range(amount):
            generator = RoadGenerator(self.bounds, self.next_seed())
            test = RoadTest(self.next_test_id(), generator.network, None, None)
            todo.append(test)
            generators[test.test_id] = generator
        yield ('init_generation', todo)
        echo = todo
        while todo:
            todo_buf = []
            for test in todo:
                generator = generators[test.test_id]
                result = generator.grow()
                if result != RoadGenerator.done:
                    todo_buf.append(test)
                else:
                    if test.network.complete_is_consistent():
                        network = test.network
                        test = self.test_from_network(network)
                        ret.append(test)
                    else:
                        generator = RoadGenerator(self.bounds, self.next_seed())
                        test = RoadTest(self.next_test_id(), generator.network, None, None)
                        todo_buf.append(test)
                        generators[test.test_id] = generator
            yield ('update_generation', echo)
            todo = todo_buf
        yield ('finish_generation', ret)

    def beg_evol_clock(self):
        self.beg_evol = datetime.datetime.now()

    def end_evol_clock(self):
        assert self.beg_evol
        ret = datetime.datetime.now() - self.beg_evol
        self.beg_evol = None
        return ret.seconds

    def start_wall_time_clock(self):
        self.wall_time = datetime.datetime.now()

    def get_wall_time_clock(self):
        assert self.wall_time
        ret = datetime.datetime.now() - self.wall_time
        return ret.seconds


    def is_new(self, candidate):
        for test in self.population:
            distance = test.distance(candidate)
            if distance < UNIQUE_THRESHOLD:
                l.info('Candidate test %s is not considered unique enough. Too similar to existing test: %s ~= %s, %s',
                       str(candidate), str(candidate), str(test), distance)
                return False
        return True

        for test in self.population:
            difference = candidate.get_path_difference(test)
            if difference < UNIQUE_THRESHOLD:
                l.info('Candidate test %s is not considered unique enough. Too similar to existing test: %s ~= %s, %s',
                       str(candidate), str(candidate), str(test), difference)
                return False
        l.info('Candidate test %s is considered new.', str(candidate))
        return True

    def evolve_suite(self, generations, time_limit=-1):
        # Start wall time clock to compute time limit
        self.start_wall_time_clock()

        # Initialise pop
        l.debug('Starting evolution clock.')

        if time_limit > 0:
            l.debug('Evolution butget is %s seconds', str(time_limit))

        self.beg_evol_clock()
        l.info('Initialising test suite population.')
        for state in self.generate_tests(self.max_pop - len(self.population)):
            if state[0] == 'finish_generation':
                self.population = state[1]
            yield state
        total_evol_time = self.end_evol_clock()
        l.debug('Paused evolution clock at: %s', total_evol_time)

        l.debug('Running initial evaluation of test suite.')
        l.debug('Using evaluator: %s', str(type(self.evaluator)))
        self.run_suite()
        evaluation = self.evaluator.evaluate_suite(self.population)
        total_eval_time = evaluation.duration
        l.debug('Paused evaluation clock at: %s', total_eval_time)

        # Also the first run counts
        if time_limit > 0 and self.get_wall_time_clock() >= time_limit:
            l.info("Enforcing time limit", time_limit,"after initial generation")
            generations = 0

        l.debug('Entering main evolution loop.')
        for _ in range(generations):
            #RoadTest.get_suite_seg_distribution(self.population, 2)
            yield ('evaluated', (self.population, evaluation, total_evol_time, total_eval_time))
            l.info('Test evolution step: %s', self.step)
            l.debug('Starting evolution clock.')
            self.beg_evol_clock()
            l.debug('Sorting population by score.')
            self.population = sorted(self.population, key=lambda x: x.score)
            if len(self.population) > self.max_pop:
                self.population = self.population[len(self.population) - self.max_pop:]
                #l.debug('Randomly shuffling population.')
                #self.rng.shuffle(self.population)
            #evaluation = self.evaluator.evaluate_suite(self.population)
            yield ('looped', self.population)

            if self.random_exp:
                for state in self.generate_tests(self.max_pop):
                    if state[0] == 'finish_generation':
                        nextgen = state[1]
            else:
                # ELITISM: NextGen always contains the BEST individual from the previous population? But best is the
                # last one?
                nextgen = [self.population[-1]]

                # INTRODUCE NEW RANDOM INDIVIDUALS IN THE POPULATION
                if self.roll_introduction():
                    test = self.generate_single_test()
                    if self.is_new(test):
                        l.info('Introducing random new test.')
                        nextgen.append(test)
                        yield ('introduce', test)

                l.debug('Selecting mates for crossover.')
                l.debug('Using selector: %s', str(type(self.selector)))
                pairs = list()
                total_pairs = len(self.population) + len(self.population)
                attempts = 0
                while len(nextgen) < self.max_pop and len(pairs) < total_pairs and attempts <= total_pairs:
                    attempts += 1
                    mom = self.selector.select(self.population)
                    l.debug('Selected mom: %s', str(mom))
                    dad = self.selector.select(self.population, ignore={mom})
                    l.debug('Selected dad: %s', str(dad))

                    pair_id = '{}x{}'.format(mom.test_id, dad.test_id)
                    pair_id2 = '{}x{}'.format(dad.test_id, mom.test_id)
                    if pair_id in pairs:
                        continue
                    if pair_id2 in pairs:
                        continue

                    pairs.append(pair_id)
                    pairs.append(pair_id2)

                    if mom == dad:
                        continue

                    children, aux = self.crossover(mom, dad)
                    if children:
                        l.debug('Produced %s children from %s x %s crossover', len(children), str(mom), str(dad))

                        l.debug('Mutating children.')
                        mutations = []
                        for child in children:
                            if self.is_new(child):
                                if self.roll_mutation(child):
                                    mutated, aux = self.mutate(child)
                                    if mutated:
                                        if self.is_new(mutated):
                                            mutations.append(mutated)
                                            test_file = c.rg.get_plots_path()
                                            test_file = os.path.join(test_file, 'test_{:06}.png'.format(mutated.test_id))
                                            #plot_test(test_file, mutated)
                                            yield ('mutated', (child, mutated, aux))
                                    else:
                                        mutations.append(child)
                                        yield ('crossedover', (mom, dad, children, aux))
                                        test_file = c.rg.get_plots_path()
                                        test_file = os.path.join(test_file, 'test_{:06}.png'.format(child.test_id))
                                        #plot_test(test_file, child)
                                else:
                                    mutations.append(child)
                                    yield ('crossedover', (mom, dad, children, aux))
                                    test_file = c.rg.get_plots_path()
                                    test_file = os.path.join(test_file, 'test_{:06}.png'.format(child.test_id))
                                    #plot_test(test_file, child)

                        for child in mutations:
                            if len(nextgen) < self.max_pop:
                                #if self.is_new(child):
                                nextgen.append(child)

            while len(nextgen) < self.max_pop:
                elite = self.population[-1]
                del self.population[-1]
                if elite not in nextgen:
                    nextgen.append(elite)

            self.population = nextgen
            self.step += 1
            total_evol_time += self.end_evol_clock()

            l.debug("Evaluating test suite after evolution step.")
            l.debug('Using evaluator: %s', str(type(self.evaluator)))
            self.run_suite()
            evaluation = self.evaluator.evaluate_suite(self.population)
            total_eval_time += evaluation.duration

            l.debug("Total Eval Time %s", str(total_eval_time))
            l.debug("Total Evol Time %s", str(total_evol_time))

            # If the time_limit is not -1 we need to enforce it
            if time_limit > 0 and self.get_wall_time_clock() >= time_limit:
                l.info("Enforcing time limit", time_limit,". Exit the evolution loop !")
                evaluation.result = TestSuiteEvaluation.RESULT_TIMEOUT
                break

        if evaluation.result == TestSuiteEvaluation.RESULT_TIMEOUT:
            yield ('timeout', (self.population, evaluation, total_evol_time, total_eval_time))
        else:
            yield ('finish_evolution', self.population)
