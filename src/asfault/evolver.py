import datetime
from time import time, sleep

import itertools

from asfault.config import *
from asfault.beamer import TestRunner, RESULT_SUCCESS, REASON_OFF_TRACK, REASON_TIMED_OUT, REASON_GOAL_REACHED
from asfault.generator import RoadGenerator, generate_networks
from asfault.plotter import *

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
        test_execution = TestExecution(self.test, random.random(), REASON_GOAL_REACHED, [],
                             random.randint(0, 5), now, now,
                             minimum_distance=0, average_distance=0.5, maximum_distance=1.0)
        test_execution.simulation_time = random.randint(10, 12)
        return test_execution


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


def determine_path(test):
    start_node = test.network.get_nodes_at(test.start)
    goal_node = test.network.get_nodes_at(test.goal)
    assert len(start_node) > 0
    assert len(goal_node) > 0
    start_node = start_node.pop()
    goal_node = goal_node.pop()
    return test.network.shortest_path(start_node, goal_node)



class DeapTestGeneration:

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

    def run_test(self, test):
        while True:
            # Test Runner is bound to the test. We need to configure the factory to return a new instance of a runner
            # configured to use the available BeamNG
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
            else:
                l.info("Already executed %s", test)

    def spawn_test(self, bounds, seed):
        network = generate_networks(bounds, [seed])[0]
        test = self.test_from_network(network)
        return test

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

    def evolve_suite(self, generations, time_limit, use_simulation_time):
        total_evaluation_time = 0
        total_evol_time = 0
        total_simulation_time = 0

        if time_limit != math.inf:
            l.info("Generation has %d seconds left", time_limit)

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
            total_simulation_time += executed_test.execution.simulation_time
            # Dumping of the population happens ONLY after the selection, meaning that if a test is executed but not selected it will not be dumped.
            yield ('test_executed', (executed_test))

            # Has the execution reached its final goal?
            if self.toolbox.stop_search(executed_test):
                l.info("The search achieved its goal. Stop.")
                # If we return at this point, the loop will not be executed
                yield ('goal_achieved', (executed_test, self.population, total_simulation_time))

            if not use_simulation_time:
                if self.get_wall_time_clock() >= time_limit:
                    l.info("Enforcing time limit")
                    # If we return at this point, the loop will not be executed
                    yield ('time_limit_reached', (self.population))
                else:
                    l.info("Remaining time %f", time_limit - self.get_wall_time_clock() )
            else:
                if total_simulation_time >= time_limit:
                    l.info("Enforcing time limit")
                    # If we return at this point, the loop will not be executed
                    yield ('time_limit_reached', (self.population))
                else:
                    l.info("Remaining time %f", time_limit - total_simulation_time)

        l.debug('Entering main evolution loop: remaining generations {}.', generations)
        natural_numbers = itertools.count()

        # Iterate over all the numbers that are less than generations
        for n in natural_numbers:

            if n > generations:
                l.info('Main evolution loop: no more generation')
                break

            l.debug('Starting evolution clock for generation %d', n)
            self.beg_evol_clock()

            l.info('Test evolution step: %s', self.step)

            l.debug('Sorting population by score (Higher score comes first')
            self.population = sorted(self.population, key=lambda ind: sum(ind.fitness.values), reverse=True)
            if len(self.population) > self.max_pop:
                l.error("Current population has more elements than expected {} instead of {}. Select best individuals",
                        len(self.population), self.max_pop)
                self.population = self.population[0:self.max_pop]

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

                if random.random() < c.ev.crossover_probability:
                    l.info("Mating (%d)Test#%d and (%d) Test#%d", mom_idx, mom.test_id, dad_idx, dad.test_id)
                    # This returns the individuals resulting from the cross-over and leaves the parents intact so
                    # they can mate again without problems

                    children, aux = self.toolbox.mate(mom, dad)

                    if children:
                        l.info('Cross-over produced %s children', len(children))

                        for idx, child in zip([mom_idx, dad_idx], children):
                            del child.fitness.values
                            offspring[idx] = child
                            l.info("Cross-over generated child {} which replaced parent({}){}", child.test_id, idx,
                                    offspring[idx].test_id)
                    else:
                        l.info('Cross-over did not produce any valid children')
                else:
                    l.debug("Did not mate ({}){} and ({}){} ", mom_idx, mom.test_id, dad_idx, dad.test_id)

            l.debug("Mutate individuals.")
            for mut_idx in range(len(offspring)):
                mutable = offspring[mut_idx]

                if random.random() < c.ev.mut_chance:
                    l.info("Mutating (%d) Test#%d ", mut_idx, mutable .test_id)

                    mutated = self.toolbox.mutate(mutable)

                    if mutated and mutated is not None:
                        l.info('Mutation produced %d', mutated.test_id)
                        del mutated.fitness.values
                        offspring[mut_idx] = mutated
                    else:
                        l.info("Mutation failed")

            next_generation.extend([child for child in offspring if not child.fitness.valid])

            # Ad this point we can notify that we generated new individuals
            yield ('finish_generation', (next_generation))

            previous_population = self.population
            self.population = next_generation
            self.step += 1
            total_evol_time += self.end_evol_clock()

            # Execute newly generated tests
            # Execute the tests one by one and evaluate their fitness
            l.info("Running newly generated tests")
            for idx, executed_test in enumerate(self.run_suite()):
                self.beg_evaluation_clock()
                l.debug("Evaluating execution of test %s", executed_test.test_id)
                fitness_values, reason = self.toolbox.evaluate(executed_test, self.population)
                executed_test.fitness.values = (fitness_values, )
                l.info('Evaluating test: {}/{} - {} - {}'.format(idx + 1, len(self.population), fitness_values, reason))
                total_evaluation_time += self.end_evaluation_clock()
                total_simulation_time += executed_test.execution.simulation_time
                # Dumping of the population happens ONLY later
                yield ('test_executed', (executed_test))

                # Has the execution reached its final goal?
                if self.toolbox.stop_search(executed_test):
                    l.debug("The search achieved its goal. Stop.")
                    yield ('goal_achieved', (executed_test, self.population, total_simulation_time))

                if not use_simulation_time:
                    if self.get_wall_time_clock() >= time_limit:
                        l.info("Enforcing time limit")
                        # Notify the "caller" about ending the generation.
                        yield ('time_limit_reached', (self.population))
                    else:
                      l.info("Remaining time %f", time_limit - self.get_wall_time_clock() )
                else:
                    if total_simulation_time >= time_limit:
                        l.info("Enforcing time limit")
                        # Notify the "caller" about ending the generation.
                        yield ('time_limit_reached', (self.population))
                    else:
                        l.info("Remaining time %f", time_limit - total_simulation_time)

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
                total_simulation_time += executed_test.execution.simulation_time
                # Dumping of the population happens ONLY later
                yield ('test_executed', (executed_test))

                # Has the execution reached its final goal?
                if self.toolbox.stop_search(executed_test):
                    l.debug("The search achieved its goal. Stop.")
                    yield ('goal_achieved', (executed_test, self.population, total_simulation_time))

                if not use_simulation_time:
                    if self.get_wall_time_clock() >= time_limit:
                        l.info("Enforcing time limit")
                        # Notify the "caller" about ending the generation.
                        yield ('time_limit_reached', (self.population))
                    else:
                        l.info("Remaining time %f", time_limit - self.get_wall_time_clock())
                else:
                    if total_simulation_time >= time_limit:
                        l.info("Enforcing time limit")
                        # Notify the "caller" about ending the generation.
                        yield ('time_limit_reached', (self.population))
                    else:
                        l.info("Remaining time %f", time_limit - total_simulation_time)

            l.info("Total Time Spent in Evaluating Tests " + str(total_evaluation_time))
            l.info("Total Time Spent in Generating Tests " + str(total_evol_time))
            l.info("Total Time Spent in Executing Tests (Simulation time) " + str(total_simulation_time))

            yield ('finish_evolution', (self.population, total_evol_time, total_evaluation_time, total_simulation_time))

        # Notify that we run all the generations and we can stop the search
        yield ('budget_limit_reached', (self.population)) #, total_evol_time, total_evaluation_time))

from asfault.mateselectors import *
