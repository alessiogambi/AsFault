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

from datetime import datetime

import re
from asfault.beamer import RESULT_SUCCESS

asfault_regex = re.compile(r".*_lanedist_.*$")
random_regex = re.compile(r".*_random_.*$")

class PopulationStats:
    SIZE = 0

    evolved_individuals = None
    padded_individuals = None

    # Overall time to evolve/generate this population
    test_generation_time = None
    # Overall time to execute the evolved tests (the others are not executed but simply copied over)
    test_execution_time = None

    def __init__(self, SIZE):
        # print("Default constructor with SIZE", SIZE)
        self.SIZE = SIZE
        self.evolved_individuals = []
        self.padded_individuals = []

    # @property ?
    def get_individuals(self):
        """
        Return all the individuals in this population
        :return:
        """
        return self.evolved_individuals + self.padded_individuals

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

    def get_test_generation_time(self):
        if self.test_generation_time is None:
            return 0
        else:
            return self.test_generation_time.total_seconds()

    def get_test_execution_time(self):
        if self.test_execution_time is None:
            return 0
        else:
            return self.test_execution_time.total_seconds()

    def append(self, individual):
        self.evolved_individuals.append(individual)

    def increase_test_execution_time_by(self, time):
        if self.test_execution_time is None:
            self.test_execution_time = time
        else:
            self.test_execution_time += time

    def increase_test_generation_time_by(self, time):
        if self.test_generation_time is None:
            self.test_generation_time = time
        else:
            self.test_generation_time += time

class LogAnalyzer:

    # Regex used to match relevant loglines (in this case, a specific IP address)

    # This identifies the start of a test execution
    test_execution_regex = re.compile(r".*Executing Test#(\d+) .*$")
    # This identifies the end of a test execution
    test_execution_stop_regex = re.compile(r".*Terminating controller process.*$")

    # This mark the beginning of the evolution step
    test_generation_start_regex = re.compile(r".*Starting evolution clock.*$")
    test_generation_end_regex = re.compile(r".*Using evaluator:.*$")

    # This mark the end of the (previous) evolution step
    evolution_regex = re.compile(r".*Test evolution step: (\d+)")

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

    def process_log(self, input_log):
        self.test_executions_folder = '/'.join([os.path.dirname(input_log), 'output/execs'])
        self.test_final_folder = '/'.join([os.path.dirname(input_log), 'output/final'])

        # Current population
        population = PopulationStats(self.POPULATION_SIZE)

        # Populations in each step of the evolution
        populations = []

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

                elif evolution_step_completed is not None:
                    # print("Evolution Step Completed", line)
                    # Check how many tests are in the population
                    if population.get_actual_size() < self.POPULATION_SIZE:
                        # Take the previous population (last element in the populations listis the first from right, i.e., -1)
                        population.pad_with(populations[-1])
                    # At this point we have the final population at evolution step and we store it
                    populations.append(population)

                    # print("Simulated evolution step", len(populations))
                    # print(" Evolved:", len(population.get_individuals()))
                    # print(" Padded", len(population.get_padded_individuals()))

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

                    # print("Adding new individual", testID, "to population with fitness", fitness)
                    population.append((testID, fitness))
                    # Timestamp is ^date[:space:]time
                    timestamp = " ".join(line.split()[0:2])
                    test_start_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')

                elif test_execution_stop_match is not None:
                    # print("Test Execution End Found", line)
                    timestamp = " ".join(line.split()[0:2])
                    test_end_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
                    duration = (test_end_time - test_start_time)
                    # print("Test executions time ", duration)
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
