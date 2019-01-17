#
# Elaborate AsFault execution logs to replay the test generation process and derive the final test suite.
#Basically:
#
#1. We know it starts out with 25 tests, all of which get executed.
#
#2. It then generates `n` new tests by crossover etc, those get executed. We can find out `n` by checking in the logs how many and which tests get executed in that generation.
#
#3. To make up for any lack of offspring, it copies over old tests from the previous gen according to fitness. So we just need to sort the previous generation by fitness and take the best `25-n` tests. (edited)


import os.path
import json
import argparse
from asfault.app import ensure_environment, DEFAULT_ENV
from asfault.tests import RoadTest, TestExecution
from asfault import config as c

from datetime import datetime, timedelta

import re
from asfault.beamer import RESULT_SUCCESS

asfault_regex = re.compile(r".*_lanedist_.*$")
random_regex = re.compile(r".*_random_.*$")


class PopulationStats:

    def __init__(self, SIZE):
        self.SIZE = SIZE
        # Overall time to evolve/generate this population
        self.test_generation_time = 0
        # Overall time to execute the evolved tests (the others are not executed but simply copied over)
        self.test_execution_time = 0
        # print("Default constructor with SIZE", SIZE)
        self.evolved_individuals = list()
        self.padded_individuals = list()
        #
        self.filtered_tests = 0
        # self.invalid_tests = 0

    def increment_filtered_tests(self):
        self.filtered_tests += 1

    def get_filtered_tests(self):
        return self.filtered_tests

    # def increment_invalid_tests(self):
    #     self.invalid_tests += 1

    def get_invalid_tests(self):
        return len(self.get_padded_individuals()) - self.filtered_tests

    def get_cumulative_fitness(self):
        """
        Compute the cumulative fitness of this population
        :return:
        """
        cumulative_fitness = 0
        for test_id, test_fitness in self.get_individuals():
            cumulative_fitness+=test_fitness
        return cumulative_fitness

    # @property ?
    def get_individuals(self):
        """
        Return all the individuals in this population
        :return:
        """
        individuals = list()
        individuals.extend(self.evolved_individuals)
        individuals.extend(self.padded_individuals)
        return individuals

    def get_padded_individuals(self):
        return self.padded_individuals

    def get_evolved_individuals(self):
        return self.evolved_individuals

    def get_size(self):
        return self.SIZE

    def get_actual_size(self):
        return len(self.get_individuals())

    def get_sorted_individuals(self):
        return sorted(self.get_individuals(), key=lambda x: x[1], reverse=True)

    def pad_with(self, another_population):
        for val in another_population.get_sorted_individuals():
            self.padded_individuals.append(val)
            if self.get_actual_size() == self.get_size():
                break

    def pad(self, individual):
        if self.get_actual_size() == self.get_size():
            return
        self.padded_individuals.append(individual)

    def pad_individuals(self, individuals):
        for individual in individuals:
            self.pad(individual)

    def get_test_generation_time(self):
        return self.test_generation_time

    def get_test_execution_time(self):
        return self.test_execution_time

    def append(self, individual):
        self.evolved_individuals.append(individual)

    def append_individuals(self, individuals):
        for individual in individuals:
            self.append(individual)

    def increase_test_execution_time_by(self, time):
        if isinstance(time, timedelta):
            self.test_execution_time += time.total_seconds()
        else:
            self.test_execution_time += time

    def increase_test_generation_time_by(self, time):
        if isinstance(time, timedelta):
            self.test_generation_time += time.total_seconds()
        else:
            self.test_generation_time += time

class LogAnalyzer:

    # Regex used to match relevant loglines (in this case, a specific IP address)

    # This identifies the start of a test execution
    test_execution_regex = re.compile(r".*Executing Test#(\d+) .*$")
    # This identifies the end of a test execution
    test_execution_stop_regex = re.compile(r".*Ending test.*$")

    # This mark the beginning of the evolution step
    test_generation_start_regex = re.compile(r".*Starting evolution clock.*$")
    test_generation_end_regex = re.compile(r".*Using evaluator:.*$")

    # This mark the end of the (previous) evolution step
    evolution_regex = re.compile(r".*Test evolution step: (\d+)")

    # This identifies non-unique tests
    filtered_test_regex = re.compile(r".*is not considered unique enough. Too similar.*$")

    # This identifies invalid tests

    # TODO Since the generation repeats the generation of tests several time, those conditions triggers more than once!
    # invalid_test_regexs = list()
    # regexs = [r".*Cross over between .* considered impossible.*$",
    #           r".*Crossing between .* is not full.*$"
    #           r".*Front and back have same sides.*$",
    #           r".*Found self-intersecting branch starting at:.*$",
    #           r".*Found a partially overlapping pair.*$",
    #           r".*Network has nodes with too many children.*$",
    #           r".*Spine starting at .* is too short.*$",
    #           r".*intersections broken.*$",
    #           r".*No two boundary segments are reachable.*$",
    #           r".*Not all branches are reachable from each other.*$",
    #           r".*Not all branches are long enough.*$",
    #           r".*Node has more than one child.*$"]

    # for regex in regexs:
    #     invalid_test_regexs.append(re.compile(regex))

    POPULATION_SIZE = -1
    GENERATION_LIMIT = -1

    evolutionStep = 0
    test_executions_folder = None
    test_final_folder = None

    def __init__(self, GENERATION_LIMIT, POPULATION_SIZE):
        ensure_environment(DEFAULT_ENV)
        self.GENERATION_LIMIT = GENERATION_LIMIT
        self.POPULATION_SIZE = POPULATION_SIZE

    def getFitnessForTest(self, testID):
        inputJSON = '/'.join([self.test_executions_folder, "test_"+testID.zfill(4)+".json"])
        # Double check that this file exists under
        if not os.path.isfile(inputJSON):
            inputJSON = '/'.join([self.test_final_folder, "test_" + testID.zfill(4) + ".json"])

        if not os.path.isfile(inputJSON):
            print("I cannot find the file ", inputJSON, "to compute the fitness value!")
            raise FileNotFoundError("File", inputJSON, "does not exist")

        with open(inputJSON) as handle:
            dictdump = json.loads(handle.read())

        test = RoadTest.from_dict(dictdump)

        if test.execution.result == RESULT_SUCCESS:
            fitness = test.execution.maximum_distance / c.ev.lane_width
            fitness = min(fitness, 1.0)
        else:
            fitness = -1

        return fitness

    def get_obe_count_for_test(self, testID):
        """
        Random does not evolve population using lanedist, but OBE count !

        :param testID:
        :return:
        """
        inputJSON = '/'.join([self.test_executions_folder, "test_"+testID.zfill(4)+".json"])
        # Double check that this file exists under
        if not os.path.isfile(inputJSON):
            inputJSON = '/'.join([self.test_final_folder, "test_" + testID.zfill(4) + ".json"])

        if not os.path.isfile(inputJSON):
            print("I cannot find the file ", inputJSON, "to compute the fitness value!")
            raise FileNotFoundError("File", inputJSON, "does not exist")

        with open(inputJSON) as handle:
            dictdump = json.loads(handle.read())

        test = RoadTest.from_dict(dictdump)
        test_execution = TestExecution.from_dict(test, dictdump["execution"])

        return len(test_execution.get_obes())

    # def match_invalid(self, line):
    #     for regex in self.invalid_test_regexs:
    #         if regex.match(line):
    #             return regex.match(line)
    #     return None

    def process_log(self, input_log):

        # Files can be either in
        # /scratch/gambi/AsFault/deepdriving/single_lanedist_0500_0750/17/output/final/test_0016.json
        # or
        # /scratch/gambi/AsFault/deepdriving/single_lanedist_0500_0750/17/.asfaultenv/output/final/test_0016.json

        self.test_executions_folder = '/'.join([os.path.dirname(input_log), 'output/execs'])
        self.test_final_folder = '/'.join([os.path.dirname(input_log), 'output/final'])

        if not os.path.exists(self.test_executions_folder):
            self.test_executions_folder = '/'.join([os.path.dirname(input_log), '.asfaultenv/output/execs'])
        if not os.path.exists(self.test_final_folder):
            self.test_final_folder = '/'.join([os.path.dirname(input_log), '.asfaultenv/output/final'])




        # Current population
        population = PopulationStats(self.POPULATION_SIZE)

        # Populations in each step of the evolution
        populations = list()

        # Open input file in 'read' mode
        with open(input_log, "r") as in_file:

            if random_regex.match(input_log):
                print("Logs for Random Execution. Use OBE count as FITNESS")
            else:
                print("Logs for AsFault Execution. Use lanedistance as FITNESS")

            evolution_step_start_time = None
            evolution_step_end_time = None
            test_start_time = None
            test_end_time = None

            for line in in_file:

                test_execution_match = self.test_execution_regex.match(line)
                test_execution_stop_match = self.test_execution_stop_regex.match(line)

                test_generation_start_match = self.test_generation_start_regex.match(line)
                test_generation_end_match = self.test_generation_end_regex.match(line)

                evolution_step_completed = self.evolution_regex.match(line)

                filtered_test_match = self.filtered_test_regex.match(line)
                # invalid_test_match = self.match_invalid(line)

                if test_generation_start_match is not None:
                    # print("Start Test Generation", line)
                    # Timestamp is ^date[:space:]time
                    timestamp = " ".join(line.split()[0:2])
                    evolution_step_start_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
                    # print("start time ", evolution_step_start_time)
                elif test_generation_end_match is not None:
                    # print("End Test Generation", line)
                    # Timestamp is ^date[:space:]time
                    timestamp = " ".join(line.split()[0:2])
                    evolution_step_end_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
                    # print("stop time", evolution_step_end_time)

                    duration = evolution_step_end_time - evolution_step_start_time
                    # print("Duration of evolution", duration)
                    population.increase_test_generation_time_by(duration)
                    # Reset counters
                    evolution_step_start_time = None
                    evolution_step_end_time = None
                elif filtered_test_match is not None:
                    print("Test is not unique enough.")
                    population.increment_filtered_tests()
                # elif invalid_test_match is not None:
                #     TODO Check that only one condition is triggered at the time !!
                #     print("Test is not valid !", invalid_test_match)
                #     population.increment_invalid_tests()
                elif evolution_step_completed is not None:
                    print("Evolution Step Completed", line)
                    # Check how many tests are in the population
                    if population.get_actual_size() < self.POPULATION_SIZE:
                        # Take the previous population (last element in the populations listis the first from right, i.e., -1)
                        population.pad_with(populations[-1])


                    if random_regex.match(input_log) and len(populations) > 0:
                        # TODO: THE FOLLOWING IS NOT WHAT WE NEED ! WE NEED TO KEEP THE BEST SUITE, NOT BUILDING THE SUITE WHICH CONTAINS THE BEST TEST CASES !
                        # print("Test Selection for Random Execution. Keep only the Best tests !")
                        # Create a set of individuals from the current and previous population
                        # all_individuals_no_duplicates = set(populations[-1].get_individuals())
                        # all_individuals_no_duplicates = all_individuals_no_duplicates.union( set(population.get_individuals() ) )
                        # Now make this a list, sort it, and take the first self.POPULATION_SIZE
                        # global_population = PopulationStats(self.POPULATION_SIZE*2)
                        # global_population.append_individuals(all_individuals_no_duplicates)
                        # Finally create the better population with the best tests
                        # better_population = PopulationStats(self.POPULATION_SIZE)
                        # better_population.append_individuals(global_population.get_sorted_individuals()[:self.POPULATION_SIZE])
                        # print("Better Population is", better_population.get_sorted_individuals())
                        # population = better_population
                        #
                        #
                        # Compare the new population against the previous one and keep the best one according to cumulative OBE
                        # since random uses OBE count as fitness, we simply sum this up
                        current_fitness = population.get_cumulative_fitness()
                        previous_fitness = populations[-1].get_cumulative_fitness()
                        if current_fitness < previous_fitness:
                            print("Old population is still better ", previous_fitness, " vs ", current_fitness)
                            # We keep the individuals of the previous population but we need to account for the timing of the
                            # new population
                            # Create a new Population nevertheless
                            merged_population = PopulationStats(self.POPULATION_SIZE)
                            # Append the individuals of the previous population
                            merged_population.append_individuals(populations[-1].get_individuals())
                            # But keep the timing for the current one, random spent time in generating and executing tests
                            merged_population.increase_test_generation_time_by(population.get_test_generation_time())
                            merged_population.increase_test_execution_time_by(population.get_test_execution_time())
                            #
                            population = merged_population
                        else:
                            print("New population is better", current_fitness, "vs", previous_fitness)

                    # At this point we have the final population at evolution step and we store it
                    populations.append(population)

                    print("Simulated evolution step", len(populations)-1)
                    print(" Evolved:", len(population.get_evolved_individuals()))
                    print(" Padded:", len(population.get_padded_individuals()))
                    print("   Invalid", population.get_invalid_tests())
                    print("   Filtered", population.get_filtered_tests())
                    print(" Generation Time", population.get_test_generation_time())
                    print(" Execution Time", population.get_test_execution_time())


                    # Reset counters and states
                    population = PopulationStats(self.POPULATION_SIZE)

                elif test_execution_match is not None:
                    # print("Test Execution Start Found", line)
                    # Extract test ID
                    testID = test_execution_match.group(1);

                    if random_regex.match(input_log):
                        # print("Logs for Random Execution. Use OBE count as FITNESS")
                        fitness = self.get_obe_count_for_test(testID)
                    else:
                        # print("Logs for AsFault Execution. Use lanedistance as FITNESS")
                        fitness = self.getFitnessForTest(testID)

                    # Timestamp is ^date[:space:]time
                    timestamp = " ".join(line.split()[0:2])
                    test_start_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')

                elif test_execution_stop_match is not None:
                    # print("Test Execution End Found", line)
                    timestamp = " ".join(line.split()[0:2])
                    test_end_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
                    duration = (test_end_time - test_start_time)
                    # print("Test executions time ", duration)
                    if duration.total_seconds() < 10:
                        print("Warning. Test", line, "lasted less than 10 seconds", duration)

                    population.append((testID, fitness))
                    # print("Adding new individual", testID, "to population with fitness", fitness, "total",
                    #       len(population.get_individuals()))

                    population.increase_test_execution_time_by(duration)

                    # Reset
                    test_start_time = None
                    test_end_time = None

                # We cap the generation since many test files are missing !
                if len(populations) >= self.GENERATION_LIMIT:
                    print("Reached GENERATION_LIMIT", self.GENERATION_LIMIT)
                    break
                    # return sorted(populations[-1], key=lambda x: x[0], reverse=False)

        print("Final population count", len(populations))
        return populations

def main():
    # Parsing CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-log', help='Input Log File')

    args = parser.parse_args()

    # State: Gen, Evaluate, Evolve
    if args.input_log is None:
        exit(1)

    la = LogAnalyzer(GENERATION_LIMIT=40, POPULATION_SIZE=25)
    la.process_log(args.input_log)


if __name__ == "__main__":
    main()
