import logging as l

from asfault.beamer import *
from asfault.evolver import *
from asfault.plotter import *
from asfault.tests import *
from asfault.mateselectors import *
from asfault.search_stoppers import *

import numpy
from asfault.mutations import *
from asfault.crossovers import *

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

    l.debug("Dumping tests: %s", ', '.join([str(test.test_id) for test in population]))

    # Dump whatever tests were generated and possibly executed so far
    for test in population:

        if test.evo_step is None and evo_step is not None:
            # Include info about their generation
            test.evo_step = evo_step

        if test.generation is None and generation is not None:
            # TODO For some weird reason this does not seem to work... Because test and executions are dump !
            test.generation = generation

        if test not in exported_tests_gen:
            l.debug("Adding %s to exported test generation", str(test.test_id))
            export_test_gen(plots_dir, tests_dir, test, render)
            exported_tests_gen.add(test)

        if test.test_id not in exported_tests_exec and test.execution:
            l.debug("Adding %s to exported test execution", str(test.test_id))
            export_test_exec(plots_dir, execs_dir, test, render)
            exported_tests_exec.add(test.test_id)


def run_deap_experiment(toolbox, factory, budget=math.inf, time_limit=math.inf, use_simulation_time=True, render=True, show=False):
    """ This function implements a generator that yields the results of each search process. In general, there might be
    different restart of the search, but one overall budget for the experiment."""

    # Setup of the experiment
    plots_dir = c.rg.get_plots_path()

    exported_tests_gen = set()
    exported_tests_exec = set()

    gen = DeapTestGeneration(toolbox, factory, cutoff=2 ** 64)

    # Counters for the budgets
    evo_step = 0
    elapsed_time = 0
    elapsed_simulation_time = 0


    plotter = None
    if show:
        plotter = EvolutionPlotter()
        plotter.start()

    # The entire process ends when the remaining budget (evolution attempts and initial population creation) is over
    # or the time limit is reached

    remaining_budget = budget
    # This is either real time or simulation time
    remaining_time = time_limit

    # WallClock time
    start_time = datetime.datetime.now()

    restarts = -1

    while True:
        # Current generation for this search process
        generation = 0

        # Update numnber of search restart
        restarts += 1

        # Update and check
        # remaining_budget -= evo_step
        #if remaining_budget <= 0:
        #    l.info("Generation budget is over.")
        #    yield ('done', ())

        # Update real execution time
        elapsed_time = datetime.datetime.now() - start_time

        # Check whether limit over real time has been reached
        if not use_simulation_time:
            remaining_time -= elapsed_time.total_seconds()
            if remaining_time <= 0:
                l.info("Time budget is over. End the experiment.")
                yield ('done', ())
        else:
            if remaining_time <= 0:
                l.info("Time budget is over. End the experiment.")
                yield ('done', ())

        # Configure the stats collector for this search process
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        # This loop is the actual search process.
        for step, data in gen.evolve_suite(remaining_budget, time_limit=remaining_time, use_simulation_time=use_simulation_time):

            evo_step += 1

            if plotter:
                updated = plotter.update((step, data))
                if updated:
                    plotter.pause()
                    if render:
                        out_file = 'step_{:09}_{}.png'.format(evo_step, step)
                        out_file = os.path.join(plots_dir, out_file)
                        save_plot(out_file, dpi=c.pt.dpi_intermediate)

            # For each execution
            if step == 'test_executed':
                l.debug("Test executed.")
                # data is a test, we create a population only for the sake of storing it...
                population = [data]
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)

            elif step == 'time_limit_reached':
                l.info("Enforcing time limit. Stopping the search")
                population = data
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)
                # This is broken
                # l.info("%s", stats.compile(population))
                yield ('done', ())

            # TODO Is this still a thing?
            elif step == 'budget_limit_reached':
                l.info("Enforcing generation limit. Stopping the search")
                population = data
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)
                # TODO This fails for some reason
                # l.info("%s", stats.compile(population))
                yield ('done', ())

            elif step == 'goal_achieved':
                l.info("Search goal achieved")
                executed_test, population, total_simulation_time = data

                # Update time limit if needed
                if use_simulation_time:
                    remaining_time -= total_simulation_time

                # Before computing the statistics of the final population we need to remove invalid individuals
                population = [test for test in population if test.fitness.valid]

                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)
                # l.info("%s", stats.compile(population))

                if c.ev.restart_search:
                    l.warning("Restart the search")
                    break
                else:
                    l.warning("Search is over")
                    yield ('done', ())

            # The test generation stage is over, evaluation is soon to begin
            elif step == 'finish_generation':
                l.info("Done generating tests for generation %s: %s", "{:03d}".format(generation), ", ".join([str(test.test_id) for test in data]))
                population = data
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)
                generation += 1

            # An evolution loop is done and a next one is about to begin so we need to log the data about execti
            elif step == 'finish_evolution':
                population, total_evol, total_eval, total_simulation_time = data
                dump_population(evo_step, generation, population, exported_tests_gen, exported_tests_exec, render)
                l.warning("Evolution step %d is finished", generation)
                l.warning("Population is : %s", ''.join([str(test.test_id) for test in population]))
                # l.warning("Statistics %s", stats.compile(population))
                generation += 1
            else:
                l.error("Unknown step %s - %s ", step, data)
                yield ('done', ())

from deap import base, creator, tools


def initIndividual(ind_class, test_generator):
    """ Ensure the creation of instances of ind_class from objects generated by test_generator"""
    random_test = test_generator()
    return wrap_test_into_individual(ind_class, random_test)


def wrap_test_into_individual(ind_class, road_test):
    if road_test and road_test is not None:
        # Note that ind_class is the constructor of the Individual class defined by creator.create
        individual = ind_class(road_test.test_id, road_test.network, road_test.start, road_test.goal)
        individual.set_path(road_test.path)
        return individual
    else:
        return None

def deap_experiment(seed, budget, factory, time_limit=-1, use_simulation_time=True, render=False, show=False):

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
    crossover = MetaCrossover()
    toolbox.register("mate", crossover.crossover)

    # MetaMutator takes car of randomly selecting mutators and retring in case they fail
    mutator = MetaMutator()

    # TODO This must be updated to retry cross-over as well
    def retry_operation(search_operator):

        def attempt_operator(*args, **kwargs):
            epsilon = 0.25
            while True:
                try:
                    offspring = search_operator(*args, **kwargs)
                    if offspring and offspring is not None:
                        return offspring
                except Exception as e:
                    l.error('Exception while creating offspring using: %s', search_operator)
                    l.exception(e)
                # Shall we give up or retry?
                failed = random.random()

                if failed < epsilon:
                    break
                else:
                    l.info("Retry the operation but increase the probability of giving up.")
                    epsilon *= 1.1

            return None

        return attempt_operator

    def wrapping(wrapped):
        def _f(*args, **kwargs):
            return wrap_test_into_individual(creator.Individual, wrapped(*args, **kwargs))
        return _f

    toolbox.register("mutate", wrapping(retry_operation(mutator.mutate)))

    assert time_limit > 0, "Time limit cannot be negative"
    if time_limit is not math.inf:
        l.info('Time limit will be enforced at: {}'.format(time_limit))

        if use_simulation_time:
            l.info("Time limit enforced on Simulation Time")
        else:
            l.info("Time limit enforced on Wallclock Time")
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
                elite = previous_population[0]
                del previous_population[0]
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
            l.info("Individual %s - %s selected as one of the best", individual.test_id, individual.fitness.values)

    def pad_with_random_from_previous(target_pop_size, current_population, previous_population):
        raise NotImplementedError()

    def pad_with_random_individuals(target_pop_size, current_population, previous_population):
        while len(current_population) < target_pop_size:
            random_ind = toolbox.individual
            del random_ind.fitness.values
            l.info("Random Individual " + str(random_ind.test_id) + " added to current population")
            current_population.append(random_ind)

    # This is mostly to enable random generation
    def replace_with_random_individuals(target_pop_size, current_population, previous_population):
        current_population.clear()
        random_population = toolbox.population(target_pop_size)
        for ind in random_population:
            del ind.fitness.values
            current_population.append(ind)
            l.info("Random Individual " + str(ind.test_id) + " added to current population")


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
    for status, data in run_deap_experiment(toolbox, factory, budget, time_limit=time_limit, use_simulation_time=use_simulation_time, render=render, show=show):
        if status == 'done':
            break
        # out_file = c.rg.get_results_path()
        # now_time = datetime.datetime.now()
        # now_time = now_time.isoformat()
        # prefix = [now_time, c.ev.evaluator, c.ev.bounds, c.ex.risk]
        # evolution = prefix + evolution
        # # Append evolution data to csv file
        # with open(out_file, 'a') as out:
        #     writer = csv.writer(out, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        #     writer.writerow(evolution)
#
# @DeprecationWarning
# def experiment(seed, budget, factory, time_limit=-1, render=False, show=False, ):
#     rng = random.Random()
#     rng.seed(seed)
#
#     if time_limit > 0:
#         l.info('Time limit will be enforced at: {}'.format(time_limit))
#     else:
#         l.info('No time limit will be enforced')
#
#     sort_pop = True
#
#     random_exp = False
#
#     # Fitness function
#     if c.ev.evaluator == 'random':
#         random_exp = True
#         evaluator = RandomEvaluator(rng)
#     elif c.ev.evaluator == 'uniqlanedist':
#         evaluator = UniqueLaneDistanceEvaluator()
#     elif c.ev.evaluator == 'max_lanedist':
#         evaluator = MaxLaneDistanceEvaluator()
#     else:
#         evaluator = LaneDistanceEvaluator()
#
#     selector = RandomMateSelector(rng)
#     # Selection
#     # if c.ev.selector == 'random':
#     #     selector = RandomMateSelector(rng)
#     #     sort_pop = False
#     # elif c.ev.selector == 'tournament':
#     #     selector = TournamentSelector(rng, 2)
#     # else:
#     #     print('Illegal value for selector: {}'.format(c.ex.selector))
#     #     sys.exit(-1)
#
#     if c.ev.estimator == 'random':
#         estimator = RandomPathEstimator(rng)
#     elif c.ev.estimator == 'length':
#         estimator = LengthEstimator()
#     else:
#         print('Illegal value for estimator: {}'.format(c.ex.estimator))
#         sys.exit(-1)
#
#     search_stopper = NeverStopSearchStopper()
#     if c.ev.search_stopper == "stop_at_obe":
#         search_stopper = StopAtObeSearchStopper()
#
#     experiment_out(rng, evaluator, selector, estimator, search_stopper,
#                    factory,
#                    sort_pop, budget, time_limit=time_limit, random_exp=random_exp, render=render, show=show)
