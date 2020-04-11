import datetime
import logging as l
import random
import sys
import math

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
            score, reason = self.evaluate_test(test, suite)
            test.score = score
            test.reason = reason

            l.info('Evaluating test: {}/{} - {} - {}'.format(idx + 1, len(suite), test.score, test.reason))

            if math.isnan(score):
                # This test did not run, so we skip the evaluation part
                continue

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
    """This evaluator does not require the test to be executed"""
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
    def __init__(self, test):
        self.test = test

    def run(self):
        now = datetime.datetime.now()
        return TestExecution(self.test, random.random(), REASON_GOAL_REACHED, [],
                             random.randint(0, 5), now, now,
                             minimum_distance=0, average_distance=0.5, maximum_distance=1.0)

    def close(self):
        pass


def gen_mock_runner_factory():
    def factory(test):
        runner = MockRunner(test)
        return runner
    return factory


class LaneDistanceEvaluator(SuiteEvaluator):
    """ Try to maximize the lane distance but cap it to 1.0 in case of OBE. We are not interested in the criticality
    of OBEs here"""

    def evaluate_test(self, test, suite):
        # Tests that where not executed cannot be assigned a score
        if test.execution is None:
            return math.nan, "Test was not executed"

        if test.execution.result == RESULT_SUCCESS:
            fitness = test.execution.maximum_distance / c.ev.lane_width
            fitness = min(fitness, 1.0)
        else:
            if test.execution.reason == REASON_OFF_TRACK:
                fitness = test.execution.maximum_distance / c.ev.lane_width
                fitness = min(fitness, 1.0)
            else:
                fitness = -1

        return fitness, test.execution.reason


class MaxLaneDistanceEvaluator(SuiteEvaluator):
    """ Try to maximize the (BOUNDED) lane distance. We are interested in the Criticality of the OBE."""
    def evaluate_test(self, test, suite):
        # Tests that where not executed cannot be assigned a score
        if test.execution is None:
            return math.nan, "Test was not executed"

        if test.execution.result == RESULT_SUCCESS:
            fitness = test.execution.maximum_distance / c.ev.lane_width
        else:
            if test.execution.reason == REASON_OFF_TRACK:
                fitness = test.execution.maximum_distance / c.ev.lane_width
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
    """Return a random evaluation of the individual. Mostly for testing purposed and to implement random generation"""
    def __init__(self, rng):
        self.rng = rng

    def evaluate_test(self, test, suite):
        if test.execution is None:
            return math.nan, "Test was not executed"

        return self.rng.random(), test.execution.reason


# TODO Not that it might be misleading to talk about test suites, when we indeed evolve test cases...
class DeapTestGeneration:
    def __init__(self):
        pass

    def __init__(self,
                 toolbox,
                 runner_factory=None,
                 **evopts):
        # DEAP framework
        self.toolbox = toolbox
        # Test runner factory
        self.runner_factory = runner_factory

        # TODO Not sure what's this...
        self.evopts = evopts

        #
        size = c.ev.bounds
        self.bounds = box(-size, -size, size, size)
        self.max_pop = c.ev.pop_size

        self.population = None

        self.test_id = 1
        self.step = 0

        self.sg_idx = 0

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
                # Execute the test
                execution = self.run_test(test)
                test.execution = execution
                # Pass the result back
                yield test

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


    # # TODO sIntroduce random individuals
    # def roll_introduction(self):
    #     return self.rng.random() <= c.ev.intro_chance

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
        # with open(sg_json, 'w') as out_file:
        # out_file.write(json.dumps(NetworkLayout.to_dict(network), indent=4, sort_keys=True))
        self.sg_idx += 1
        # plot_network(sg_file, network)
        if candidates:
            for start, goal in candidates:
                # l.info(sg_file)
                l.info('Checking candidate: (%s, %s), %s/%s', start, goal, candidate_idx, len(candidates))
                candidate_idx += 1
                paths = network.all_paths(start, goal)
                # paths = network.all_shortest_paths(start, goal)
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


    def beg_evol_clock(self):
        self.beg_evol = datetime.datetime.now()

    def end_evol_clock(self):
        assert self.beg_evol
        ret = datetime.datetime.now() - self.beg_evol
        self.beg_evol = None
        return ret.seconds

    def beg_evaluation_clock(self):
        self.beg_evaluation = datetime.datetime.now()

    def end_evaluation_clock(self):
        assert self.beg_evaluation
        ret = datetime.datetime.now() - self.beg_evaluation
        self.beg_evaluation = None
        return ret.seconds

    def start_wall_time_clock(self):
        self.wall_time = datetime.datetime.now()

    def get_wall_time_clock(self):
        assert self.wall_time
        ret = datetime.datetime.now() - self.wall_time
        return ret.seconds

    def check_stopping_condition(self):
        return False

    def evolve_suite(self, generations, time_limit):
        # TODO Check that 1+1EA is still possible
        # TODO Check that RANDOM is still possible

        total_evaluation_time = 0
        total_evol_time = 0

        # Start wall time clock to compute time limit
        self.start_wall_time_clock()

        # Initialise pop
        l.debug('Starting evolution clock.')
        self.beg_evol_clock()
        l.debug('Initialising test suite population.')
        self.population = self.toolbox.population(self.max_pop)

        total_evol_time += self.end_evol_clock()
        l.debug('Paused evolution clock at: %s', total_evol_time)

        # Notify caller about progress
        yield ('finish_generation', self.population)

        # Evaluate the initial population
        l.debug('Running initial evaluation of test suite.')

        # Execute the tests one by one and evaluate their fitness
        for idx, executed_test in enumerate(self.run_suite()):

            self.beg_evaluation_clock()
            l.debug("Evaluating execution of test %s", executed_test.test_id)
            fitness_values, reason = self.toolbox.evaluate(executed_test, self.population)
            executed_test.fitness.values = (fitness_values,)
            l.info('Evaluating test: {}/{} - {} - {}'.format(idx + 1, len(self.population), fitness_values, reason))
            total_evaluation_time += self.end_evaluation_clock()

            # Has the execution reached its final goal?
            if self.toolbox.stop_search(executed_test):
                l.info("The search achieved its goal. Stop.")
                # If we return at this point, the loop will not be executed
                yield ('goal_achieved', (executed_test, self.population))

            if self.get_wall_time_clock() >= time_limit:
                l.info("Enforcing time limit %d", time_limit)
                # If we return at this point, the loop will not be executed
                yield ('time_limit_reached', (self.population))

        l.debug('Entering main evolution loop: remaining generations {}.', generations)
        for _ in range(generations):
            # TODO How to implement random search?

            # TODO Notify the caller about all the tests being evaluated to store info ?
            # yield ('evaluated', (self.population, evaluation, total_evol_time, total_eval_time))

            l.debug('Starting evolution clock.')
            self.beg_evol_clock()

            l.info('Test evolution step: %s', self.step)

            l.debug('Sorting population by score (Higher score comes first')
            self.population = sorted(self.population, key=lambda ind: sum(ind.fitness.values), reverse=True)
            if len(self.population) > self.max_pop:
                l.error("Current population has more elements than expected {} instead of {}. Select best individuals",
                        len(self.population), self.max_pop)
                self.population = self.population[0:self.max_pop]
            # TODO No idea why I need to notify about this ...
            # yield ('looped', self.population)

            best_individual = self.population[0]
            l.debug("Promote best individual {}", best_individual.test_id)
            next_generation = [best_individual]

            # SELECT INDIVIDUALS
            l.debug('Select individuals.')
            offspring = self.toolbox.select(self.population, self.max_pop)

            l.debug('Mate individuals.')
            # Since our search operators generates new individuals, we cannot clone the offspring to mate and and
            # mutate them, instead we keep track of their index and replace the objects as we move on
            for mom_idx, dad_idx in zip(range(len(offspring))[::2], range(len(offspring))[1::2]):

                mom = offspring[mom_idx]
                dad = offspring[dad_idx]

                l.debug("Selected ({}){} and ({}){} ", mom_idx, mom.test_id, dad_idx, dad.test_id)

                if mom == dad:
                    l.debug("Same individual. Move on")
                    continue

                # TODO This should be replaces with CXP
                if random.random < c.crossover_probability:
                    l.debug("Mating ({}){} and ({}){} ", mom_idx, mom.test_id, dad_idx, dad.test_id)
                    # This returns the individuals resulting from the cross-over and leaves the parents intact so
                    # they can mate again without problems

                    children, aux = self.toolbox.mate(mom, dad)

                    if children:
                        l.debug('Cross-over produced %s children', len(children))

                    for idx, child in zip([mom_idx, dad_idx], children):
                        del child.fitness.values
                        offspring[idx] = child
                        l.debug("Cross-over generated child {} which replaced parent({}){}", child.test_id, idx,
                                offspring[idx].test_id)
                else:
                    l.debug("Did not mate ({}){} and ({}){} ", mom_idx, mom.test_id, dad_idx, dad.test_id)

            l.debug("Mutate individuals.")
            for mut_idx in range(len(offspring)):
                mutable = offspring[mut_idx]

                if random.random() < c.ev.mut_chance:
                    l.debug("Mutating ({}){} ", mut_idx, mutable .test_id)

                    mutated, aux = self.toolbox.mutate(mutable)

                    if mutated:
                        l.debug('Mutation produced {}', mutated.test_id)
                        del mutated.fitness.values
                        offspring[mut_idx] = mutated

            next_generation.extend([child for child in offspring if not child.fitness.valid])

            previous_population = self.population
            self.population = next_generation
            self.step += 1
            total_evol_time += self.end_evol_clock()

            # Execute newly generated tests
            # Execute the tests one by one and evaluate their fitness
            l.debug("Running newly generated tests")
            for idx, executed_test in enumerate(self.run_suite()):
                self.beg_evaluation_clock()
                l.debug("Evaluating execution of test %s", executed_test.test_id)
                fitness_values, reason = self.toolbox.evaluate(executed_test, self.population)
                executed_test.fitness.values = (fitness_values, )
                l.info('Evaluating test: {}/{} - {} - {}'.format(idx + 1, len(self.population), fitness_values, reason))
                total_evaluation_time += self.end_evaluation_clock()

                # Has the execution reached its final goal?
                if self.toolbox.stop_search(executed_test):
                    l.debug("The search achieved its goal. Stop.")
                    return ('goal_achieved', (executed_test, self.population))

                if self.get_wall_time_clock() >= time_limit:
                    l.info("Enforcing time limit %d", time_limit)
                    # Notify the "caller" about ending the generation.
                    return ('time_limit_reached', (self.population))

            # Combine the next_generation with the previous population
            self.toolbox.merge_populations(self.population, previous_population)

            # This might create new random tests which are not yet evaluated.... So we need to execute them
            for idx, executed_test in enumerate(self.run_suite()):
                self.beg_evaluation_clock()
                l.debug("Evaluating execution of test %s", executed_test.test_id)
                fitness_values, reason = self.toolbox.evaluate(executed_test, self.population)
                executed_test.fitness.values = (fitness_values, )
                l.info('Evaluating test: {}/{} - {} - {}'.format(idx + 1, len(self.population), fitness_values, reason))
                total_evaluation_time += self.end_evaluation_clock()

                # Has the execution reached its final goal?
                if self.toolbox.stop_search(executed_test):
                    l.debug("The search achieved its goal. Stop.")
                    return ('goal_achieved', (executed_test, self.population))

                if self.get_wall_time_clock() >= time_limit:
                    l.info("Enforcing time limit %d", time_limit)
                    # Notify the "caller" about ending the generation.
                    return ('time_limit_reached', (self.population))

            # TODO What about test EXECUTION time?
            l.debug("Total Time Spent in Evaluating Tests {}", str(total_evaluation_time))
            l.debug("Total Time Spent in Generating Tests {}", str(total_evol_time))

            # Notify that we completed an evolution round
            yield ('finish_evolution', (self.population))

        # Notify that we run all the generations and we can stop the search
        yield ('budget_limit_reached', (self.population)) #, total_evol_time, total_evaluation_time))

from asfault.selectors import *
from asfault.estimators import *
from asfault.search_stoppers import *

@DeprecationWarning
class TestSuiteGenerator:

    def __init__(self,
                 rng=random.Random(),
                 evaluator=StructureEvaluator(),
                 selector=TournamentSelector(3),
                 estimator=TurnAndLengthEstimator(),
                 search_stopper=NeverStopSearchStopper(),
                 runner_factory=None,
                 sort_pop=True,
                 plotter=None,
                 **evopts):
        self.rng = rng
        self.evaluator = evaluator
        self.selector = selector
        self.estimator = estimator
        self.search_stopper = search_stopper
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
                # Execute the test
                execution = self.run_test(test)
                test.execution = execution
                # Pass the result back to enable
                yield execution


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

        l.info('Candidate test %s is considered new.', str(candidate))
        return True

    def check_stopping_condition(self):
        return False

    # "Interacting evolution"
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

        restart_search = False

        ### UGLY: THIS SHOULD BE REFACTORED TO FIT IN A METHOD OR SOMETHING
        # Instead of executing everything AND then check conditions run_suite yields the partial result and we decide
        # what to do next.
        for execution in self.run_suite():

            # Has the execution reached its final goal?
            if self.search_stopper.stopping_condition_met(execution):
                l.info("Search achieved its goal. Restart the search...")
                restart_search = True
                yield ('goal_achieved', (execution, self.population))
                break

            # Is the time limit reached?
            if time_limit > 0 and self.get_wall_time_clock() >= time_limit:
                l.info("Enforcing time limit %d after initial generation", time_limit)
                # Notify the "caller" about ending the generation.
                yield ('time_limit_reached', (self.population))
                # We need to evaluate the tests generate so far
                break


        # if not restart_search:
        if True:
            # Compute fitness function of the individuals. Not really necessary
            evaluation = self.evaluator.evaluate_suite(self.population)
            total_eval_time = evaluation.duration
            l.debug('Paused evaluation clock at: %s', total_eval_time)

            if time_limit > 0 and self.get_wall_time_clock() >= time_limit:
                return

        l.debug('Entering main evolution loop.')
        for _ in range(generations):

            l.debug('Starting evolution clock.')
            self.beg_evol_clock()

            # EVOLUTION OR RESTART THOSE GENERETE THE NEXT GEN
            if self.random_exp or restart_search:
                if restart_search:
                    l.info("Restarting the search")
                    restart_search = False

                for state in self.generate_tests(self.max_pop):
                    if state[0] == 'finish_generation':
                        nextgen = state[1]
            else:
                # RoadTest.get_suite_seg_distribution(self.population, 2)
                yield ('evaluated', (self.population, evaluation, total_evol_time, total_eval_time))

                l.info('Test evolution step: %s', self.step)
                l.debug('Sorting population by score.')
                self.population = sorted(self.population, key=lambda x: x.score)
                if len(self.population) > self.max_pop:
                    self.population = self.population[len(self.population) - self.max_pop:]
                yield ('looped', self.population)

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

                # SELECT INDIVIDUALS
                l.debug('Selecting mates for crossover.')
                l.debug('Using selector: %s', str(type(self.selector)))
                pairs = list()
                total_pairs = len(self.population) + len(self.population)
                attempts = 0

                # PATCH TO ENABLE 1+1EA
                #   pop has 1 individual, we mutate it (always), and select the best between original and mutation
                if self.max_pop == 1:
                    l.info("1+1 EA")
                    resident = self.population[0]
                    mutated, aux = self.mutate(resident)
                    # Mutation might go wrong
                    if mutated:
                        if self.is_new(mutated):
                            # TODO Not sure we actually need test_file
                            test_file = c.rg.get_plots_path()
                            test_file = os.path.join(test_file, 'test_{:06}.png'.format(mutated.test_id))
                            yield ('mutated', (resident, mutated, aux))

                            # At this point we add this into nextGen and move on to the evaluation part
                            nextgen.append(mutated)
                        else:
                            l.warning("Mutated individual %s is NOT considered new/novel", mutated.test_id)
                    else:
                        l.info("Invalid mutation %s", aux)
                else:
                    # This works Only if there are more than 1? INDIVIDUALS
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

                        # CROSSOVER
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
                                    nextgen.append(child)

            # ELITISM: PAD NEW GENERATION WITH BEST INDIVIDUALS
            while len(nextgen) < self.max_pop:
                elite = self.population[-1]
                del self.population[-1]
                if elite not in nextgen:
                    nextgen.append(elite)

            # EVALUATION of the new population (requires to set self.population) unless search must be restarted
            # Note that nextgen ALWAYS contains the BEST individual of the current population
            self.population = nextgen
            self.step += 1
            total_evol_time += self.end_evol_clock()

            # This executes ALL the tests in one shot
            for execution in self.run_suite():
                # Has the execution reached its final goal?
                if self.search_stopper.stopping_condition_met(execution):
                    l.info("Search achieved its goal. Restart the search...")
                    restart_search = True
                    yield ('goal_achieved', (execution, self.population))
                    break

                # Is the time limit reached?
                if time_limit > 0 and self.get_wall_time_clock() >= time_limit:
                    l.info("Enforcing time limit %d after initial generation", time_limit)
                    # Notify the "caller" about ending the generation.
                    yield ('time_limit_reached', (self.population))
                    # We need to evaluate whatever tests we did created
                    break

            # if not restart_search:
            if True:
                l.info("Evaluating test suite after evolution step.")
                l.debug('Using evaluator: %s', str(type(self.evaluator)))

                evaluation = self.evaluator.evaluate_suite(self.population)
                total_eval_time += evaluation.duration

                if time_limit > 0 and self.get_wall_time_clock() >= time_limit:
                    return
                
                # AT THIS POINT WE CAN COMPARE OLD AND NEW TO DECIDE WHICH INDIVIDUALS TO KEEP AROUND - This will be
                # overwritten by restart_search
                if self.max_pop == 1:
                    # At this point we have a population made of at most 2 individuals, and we pick the one which has
                    # the higher score. Note that using LaneDist once we get an obe score goes to 1, no matter how many
                    # OBEs the test caused.
                    # If we keep the original one, we might end up in trying out all the mutations cyclically, while if we
                    # select always the new one, we are piling up mutations. A third option, would be to keep the test
                    # which achieved the most OBEs... Which eventually results in mutating the best one over and over...
                    #
                    # Hence I decide to keep the new ones to favor the exploration no matter the number of OBEs.
                    l.info("1+1EA: selecting best individual between %s", ', '.join(['--'.join([str(test.test_id), str(test.score)]) for test in self.population]))
                    # Ensures that only one element survives
                    self.population = sorted(self.population, key=lambda t: t.score, reverse=True)[0:1]
                    l.info("1+1EA: Best individual is %s", self.population[-1].test_id)

                l.debug("Total Eval Time %s", str(total_eval_time))
                l.debug("Total Evol Time %s", str(total_evol_time))

                if evaluation.result == TestSuiteEvaluation.RESULT_TIMEOUT:
                    yield ('timeout', (self.population, evaluation, total_evol_time, total_eval_time))
                else:
                    yield ('finish_evolution', self.population)
