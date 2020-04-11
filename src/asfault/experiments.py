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
from asfault.selectors import *
from asfault.search_stoppers import *
from asfault.estimators import *


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


def dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render):
    execs_dir = c.rg.get_execs_path()
    plots_dir = c.rg.get_plots_path()
    tests_dir = c.rg.get_tests_path()

    # Dump whatever tests were generated and possibly executed so far
    for test in population:

        if test.evo_step is None and evo_step is not None:
            # Include info about their generation
            test.evo_step = evo_step

        if test.generation is None and generation is not None:
            # TODO For some weird reason this does not seem to work... Because test and executions are dump !
            test.generation = generation

        if test not in exported_tests_gen:
            export_test_gen(plots_dir, tests_dir, test, render)
            exported_tests_gen.add(test)

        if test.test_id not in exported_tests_exec and test.execution:
            export_test_exec(plots_dir, execs_dir, test, render)
            exported_tests_exec.add(test.test_id)

@DeprecationWarning
def run_experiment(rng, evaluator, selector, estimator, search_stopper, factory, sort_pop, budget, time_limit=-1, random_exp=False, render=True, show=False):
    plots_dir = c.rg.get_plots_path()

    exported_tests_gen = set()
    exported_tests_exec = set()

    gen = TestSuiteGenerator(rng, evaluator, selector, estimator, search_stopper, factory,
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

        if step == 'time_limit_reached':
            l.warning("Enforcing the time limit. Stopping the search")
            population = data
            dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)


        if step == 'goal_achieved':
            l.warning("GOAL ACHIEVED !!!")
            execution, population = data
            dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)

            # Reset the generation counter
            generation = 0

        if step == 'finish_generation' or step == 'looped':
            # Dump the population to log file
            l.warning("GENERATION %s: %s", "{:03d}".format(generation), ", ".join([str(test.test_id) for test in data]))
            population = data
            dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)

        if step == 'finish_evolution':
            l.warning("FINAL TEST SUITE: %s", ", ".join([str(test.test_id) for test in data]))
            final_path = c.rg.get_final_path()
            population = data
            dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)

        if step == 'evaluated':

            population, evaluation, total_evol, total_eval = data

            dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)

            fitness = evaluation.score
            oob = evaluation.oob
            timeouts = evaluation.timeouts
            diversity = evaluation.coverage
            min_test = get_worst_test(population).test_id
            max_test = get_best_test(population).test_id

            evo.extend([generation, fitness, oob, timeouts,
                        diversity, min_test, max_test])
            evo.extend([total_evol, evaluation.total_coverage])

            # This should be incremented after the execution not before...
            generation += 1

            yield evo


def run_deap_experiment(toolbox, factory, budget, time_limit=math.inf, render=True, show=False):
    plots_dir = c.rg.get_plots_path()

    exported_tests_gen = set()
    exported_tests_exec = set()

    gen = DeapTestGeneration(toolbox, factory, cutoff=2 ** 64)

    generation = 0
    evo_step = 0
    elapsed_time = 0

    plotter = None
    if show:
        plotter = EvolutionPlotter()
        plotter.start()

    # The entire process ends when the remaining budget (evolution attempts and initial population creation) is over
    # or the time limit is reached
    remaining_budget = budget
    remaining_time = time_limit
    while True:

        remaining_budget -= generation
        remaining_time -= elapsed_time

        # Current generation
        generation = 0

        if remaining_budget <= 0:
            l.info("Generation budget is over.")
            return None

        # This is the search cycle. Every time it restarts we decrease the generation budget
        for step, data in gen.evolve_suite(remaining_budget, time_limit=time_limit):
            evo_step += 1

            if plotter:
                updated = plotter.update((step, data))
                if updated:
                    plotter.pause()
                    if render:
                        out_file = 'step_{:09}_{}.png'.format(evo_step, step)
                        out_file = os.path.join(plots_dir, out_file)
                        save_plot(out_file, dpi=c.pt.dpi_intermediate)

            if step == 'time_limit_reached':
                l.warning("Enforcing the time limit. Stopping the search")
                population = data
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)
                return None

            if step == 'budget_limit_reached':
                l.warning("Enforcing the budget limit. Stopping the search")
                population = data
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)
                return None

            if step == 'goal_achieved':
                l.warning("GOAL ACHIEVED")
                _, population = data
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)

                if c.ev.restart_search:
                    l.warning("RESTART THE SEARCH")
                    # Exit the search loop and restart the search
                    # TODO Maybe we need to yield something back?
                    break
                else:
                    l.warning("SEARCH FINISHED: %s", ", ".join([str(test.test_id) for test in data]))
                    return None

            # The test generation stage is over, evaluation is soon to begin
            if step == 'finish_generation':
                l.warning("Done test generation %s: %s", "{:03d}".format(generation), ", ".join([str(test.test_id) for test in data]))
                population = data
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)
                generation += 1

            # TODO Whaet's this?
            # Not sure what's looped?
            if step == 'looped':
                l.warning("LOOPED GENERATION %s: %s", "{:03d}".format(generation), ", ".join([str(test.test_id) for test in data]))
                population = data
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)

            # The evolution loop is done and a next one is about to begin
            if step == 'finish_evolution':
                # TODO This is wrong?
                l.warning("EVOLUTION FINISHED: %s", ", ".join([str(test.test_id) for test in data]))
                # final_path = c.rg.get_final_path()
                population = data
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)
                generation += 1

            # TODO What's this? the status of the things after we evaluated the individuals?
            if step == 'evaluated':
                population, evaluation, total_evol, total_eval = data
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)
                fitness = evaluation.score
                oob = evaluation.oob
                timeouts = evaluation.timeouts
                diversity = evaluation.coverage
                min_test = get_worst_test(population).test_id
                max_test = get_best_test(population).test_id
                # Encapsulate data to return to the outer caller printing to file..
                evo = list()
                evo.extend([generation, fitness, oob, timeouts,
                            diversity, min_test, max_test])
                evo.extend([total_evol, evaluation.total_coverage])
                # This should be incremented after the execution not before...
                yield evo

@DeprecationWarning
def experiment_out(rng, evaluator, selector, estimator, search_stopper, factory, sort_pop, budget, time_limit=-1, random_exp=False, render=False, show=False):
    out_file = c.rg.get_results_path()
    with open(out_file, 'w'):
        pass
    for evolution in run_experiment(rng, evaluator, selector, estimator, search_stopper, factory, sort_pop, budget,
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


from deap import base, creator, tools

def initIndividual(ind_class, test_generator):
    """ Ensure the creation of instances of ind_class from objects generated by test_generator"""
    random_test = test_generator()
    # Note that ind_class is the constructor of the Individual class defined by creator.create
    random_individual = ind_class(random_test.test_id, random_test.network, random_test.start, random_test.goal)
    random_individual.set_path(random_test.path)

    return random_individual


def deap_experiment(seed, budget, factory, time_limit=-1, render=False, show=False, ):

    # Ensure we use the provided seed for repeatability
    random.seed(seed)

    # So far we only one fitness
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # Define a class named Individuals that extends RoadTest with the attribute fitness
    creator.create("Individual", RoadTest, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # We use a RoadTestFactory to create (or load) RoadTests from a map of the expected size
    test_factory = RoadTestFactory(c.ev.bounds)

    # Create a random instance of RoadTest using 'test_factory' and wrap it as class of type creator.Individual
    toolbox.register("random_individual", initIndividual, creator.Individual, test_factory.generate_random_test)

    # Create an entire population represented as list made of random individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.random_individual)

    # Register the cross-over function. MetaCrossover takes care of selecting the right crossover and retry its
    # application automatically. TODO Better name maybe?
    crossover = crossovers.MetaCrossover()
    toolbox.register("mate", crossover.crossover)

    # MetaMutator takes car of randomly selecting mutators and retring in case they fail
    mutator = mutations.MetaMutator()
    toolbox.register("mutate", mutator.mutate)

    assert time_limit > 0, "Time limit cannot be negative"
    if time_limit is not math.inf:
        l.info('Time limit will be enforced at: {}'.format(time_limit))
    else:
        l.info('No time limit will be enforced')

    # Register the fitness function
    if c.ev.evaluator == 'unique_lanedist':
        evaluator = UniqueLaneDistanceEvaluator()
    elif c.ev.evaluator == 'max_lanedist':
        evaluator = MaxLaneDistanceEvaluator()
    elif c.ev.evaluator == 'lanedist':
        evaluator = LaneDistanceEvaluator()
    # elif c.ev.evaluator == 'curvature_entropy':
    #     evaluator = CurvatureEntropyEvaluator()
    # elif c.ev.evaluator == 'spped_entropy':
    #     evaluator = SpeedEntropyEvaluator()
    else:
        raise Exception("Missing Fitness Function")

    # def evaluate_test(self, test, suite):
    toolbox.register("evaluate", evaluator.evaluate_test)

    # TODO Not sure this is really needed !!
    # Not sure this is needed
    # toolbox.register("evaluate_population", evaluator.evaluate_suite)

    # Selection - We use the original selection mechanisms for the moment
    if c.ev.selector == 'random':
        # selector = RandomMateSelector()
        toolbox.register("select", tools.selRandom)
    elif c.ev.selector == 'tournament':
        # selector = TournamentSelector(tourney_size=2)
        toolbox.register("select", tools.selTournament, tournsize=2)
    else:
        raise Exception("Missing Selector Function")


    # The following functions compute score metrics on the path/test in order to filter them out. This is an advanced
    # feature that we will not use at the moment
    # Not sure what do they estimate... probably some feature of the individuals?
    # Optional ?
    # TODO Those act on the PATH...
    # if c.ev.estimator == 'random':
    #     estimator = RandomPathEstimator()
    # elif c.ev.estimator == 'length':
    #     estimator = LengthEstimator()
    # else:
    #     raise Exception("Missing Estimator Function")
    #
    # TODO Not sure what do we estimate
    # toolbox.register("estimate", selector.select)

    # Define the condition upon which the search should end
    # Is time limit to be considered here?
    search_stopper = NeverStopSearchStopper()
    if c.ev.search_stopper == "stop_at_obe":
        search_stopper = StopAtObeSearchStopper()
    # This requires an individual to be executed...
    toolbox.register("stop_search", search_stopper.stopping_condition_met)

    # TODO Refactor the following code !
    # As a last step we need to combine the new generated tests and the past population
    def pad_with_best_individuals(target_pop_size, current_population, previous_population):
        """ Pad current population with the best individuals from the previous population. This is the original
        approach implemented by AsFault"""
        if len(current_population) < target_pop_size:
            l.debug("Pad current population with best individuals of previous population")
            # This should be already sorted
            previous_population = sorted(previous_population, key=lambda ind: sum(ind.fitness.values), reverse=True)

            while len(current_population) < target_pop_size:
                elite = previous_population.population[0]
                del previous_population.population[0]
                if elite not in current_population:
                    l.debug("Promoting {} to current population", elite.test_id)
                    current_population.append(elite)

    def select_best_individuals(target_pop_size, current_population, previous_population):
        population_made_of_best_individuals = list()
        population_made_of_best_individuals.extend(current_population)
        population_made_of_best_individuals.extend(previous_population)
        # Remove duplicates
        population_made_of_best_individuals = list(set(population_made_of_best_individuals))
        # Sort by fitness
        population_made_of_best_individuals = sorted(population_made_of_best_individuals , key=lambda ind: sum(ind.fitness.values), reverse=True)
        # Copy over only the best
        current_population.clear()
        current_population.extend(population_made_of_best_individuals[0:target_pop_size])
        # Remove duplicates?
        for individual in current_population:
            l.debug("Individual {} - {} selected as one of the best", individual.test_id, individual.fitness.values)

    def pad_with_random_from_previous(target_pop_size, current_population, previous_population):
        raise NotImplementedError()

    def pad_with_random_individuals(target_pop_size, current_population, previous_population):
        while len(current_population) < target_pop_size:
            random_ind = toolbox.individual
            del random_ind.fitness.values
            l.debug("Random Individual {} added to current population", random_ind.test_id)
            current_population.append(random_ind)

    # This is mostly to enable random generation
    def replace_with_random_individuals(target_pop_size, current_population, previous_population):
        current_population.clear()
        random_population = toolbox.population(target_pop_size)
        for ind in random_population:
            del ind.fitness.values
            current_population.append(ind)


    if c.ev.pop_merger == 'pad_with_random':
        toolbox.register("merge_populations", pad_with_random_individuals, c.ev.pop_size)
    elif c.ev.pop_merger == 'take_best':
        toolbox.register("merge_populations", select_best_individuals, c.ev.pop_size)
    elif c.ev.pop_merger == 'pad_with_best':
        toolbox.register("merge_populations", pad_with_best_individuals, c.ev.pop_size)
    elif c.ev.pop_merger == 'replace_with_random':
        toolbox.register("merge_populations", replace_with_random_individuals, c.ev.pop_size)
    else:
        raise Exception("Merge populations not defined")

    # Execute the experiments using run_experiment and output the results
    out_file = c.rg.get_results_path()
    # What's this? touch? or a method to wipe out the file?
    with open(out_file, 'w'):
        pass

    # This report the result of a single evolution step to be logged so fitness and such can be seen
    for evolution in run_deap_experiment(toolbox, factory, budget, time_limit=time_limit, render=render, show=show):
        out_file = c.rg.get_results_path()
        now_time = datetime.datetime.now()
        now_time = now_time.isoformat()
        prefix = [now_time, c.ev.evaluator, c.ev.bounds, c.ex.risk]
        evolution = prefix + evolution
        # Append evolution data to csv file
        with open(out_file, 'a') as out:
            writer = csv.writer(out, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(evolution)

@DeprecationWarning
def experiment(seed, budget, factory, time_limit=-1, render=False, show=False, ):
    rng = random.Random()
    rng.seed(seed)

    if time_limit > 0:
        l.info('Time limit will be enforced at: {}'.format(time_limit))
    else:
        l.info('No time limit will be enforced')

    sort_pop = True

    random_exp = False

    # Fitness function
    if c.ev.evaluator == 'random':
        random_exp = True
        evaluator = RandomEvaluator(rng)
    elif c.ev.evaluator == 'uniqlanedist':
        evaluator = UniqueLaneDistanceEvaluator()
    elif c.ev.evaluator == 'max_lanedist':
        evaluator = MaxLaneDistanceEvaluator()
    else:
        evaluator = LaneDistanceEvaluator()

    selector = RandomMateSelector(rng)
    # Selection
    # if c.ev.selector == 'random':
    #     selector = RandomMateSelector(rng)
    #     sort_pop = False
    # elif c.ev.selector == 'tournament':
    #     selector = TournamentSelector(rng, 2)
    # else:
    #     print('Illegal value for selector: {}'.format(c.ex.selector))
    #     sys.exit(-1)

    if c.ev.estimator == 'random':
        estimator = RandomPathEstimator(rng)
    elif c.ev.estimator == 'length':
        estimator = LengthEstimator()
    else:
        print('Illegal value for estimator: {}'.format(c.ex.estimator))
        sys.exit(-1)

    search_stopper = NeverStopSearchStopper()
    if c.ev.search_stopper == "stop_at_obe":
        search_stopper = StopAtObeSearchStopper()

    experiment_out(rng, evaluator, selector, estimator, search_stopper,
                   factory,
                   sort_pop, budget, time_limit=time_limit, random_exp=random_exp, render=render, show=show)
