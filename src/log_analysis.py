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
from asfault.tests import RoadTest
from asfault import config as c

import re
from asfault.beamer import RESULT_SUCCESS

class LogAnalyzer:

    # Regex used to match relevant loglines (in this case, a specific IP address)
    execution_regex = re.compile(r".*Executing Test#(\d+) .*$")
    evolution_regex = re.compile(r".*Test evolution step: (\d+)")

    POPULATION_SIZE = 25
    evolutionStep = 0
    test_executions_folder = None
    test_final_folder = None

    def __init__(self):
        ensure_environment(DEFAULT_ENV)

    def getFitnessForTest(self, testID):
        inputJSON = '/'.join([self.test_executions_folder, "test_"+testID.zfill(4)+".json"])
        # Double check that this file exists under
        if not os.path.isfile(inputJSON):
            inputJSON = '/'.join([self.test_final_folder, "test_" + testID.zfill(4) + ".json"])

        with open(inputJSON) as handle:
            dictdump = json.loads(handle.read())

        test = RoadTest.from_dict(dictdump)

        if test.execution.result == RESULT_SUCCESS:
            fitness = test.execution.maximum_distance / c.ev.lane_width
            fitness = min(fitness, 1.0)
        else:
            fitness = -1

        return fitness

    def process_log(self, input_log):
        self.test_executions_folder = '/'.join([os.path.dirname(input_log), 'output/execs'])
        self.test_final_folder = '/'.join([os.path.dirname(input_log), 'output/final'])

        # Initial state
        state = "Execution"

        # Initial population
        population = []

        # Population in each step of the evolution
        populations = []

        # Open input file in 'read' mode
        with open(input_log, "r") as in_file:
            # Loop over each log line
            for line in in_file:

                test_execution_match = self.execution_regex.match(line)
                test_evolution_match = self.evolution_regex.match(line)

                if state == "Execution":
                    if test_execution_match is not None:
                        # Extract test ID
                        testID = test_execution_match.group(1)
                        fitness = self.getFitnessForTest(testID)
                        # print("line", line)
                        # print("Adding ", testID, "to population")
                        population.append((testID, fitness))

                    if test_evolution_match is not None:
                        state = "Evolution"
                        # Check how many tests are in the population
                        if len(population) < 25:
                            # print("***** Missing ", (25 - len(population)), " tests.")
                            # Take the last element in the populations list, which is the first from right, i.e., -1
                            sorted_population = sorted(populations[-1], key=lambda x: x[1], reverse=True)
                            # print("Sorted population: ", sorted_population);
                            for idx, val in enumerate(sorted_population):
                                # print("Adding ", idx, val, "to population")
                                population.append(val)
                                if len(population) == 25:
                                    break

                        populations.append(population)
                        print("Simulated evolution step", len(populations))

                if state == "Evolution":
                    if test_execution_match is not None:
                        # Create a new population
                        population = []
                        # Extract test ID
                        testID = test_execution_match.group(1);
                        fitness = self.getFitnessForTest(testID)
                        # print("line", line)
                        # print("Adding ", testID, "to population")
                        population.append((testID, fitness))
                        # print(len(population))
                        state = "Execution"

        # At this point the latest element of populations contains the final test suite
        # print("FINAL TEST SUITE:")
        # # Present order by testID
        # for idx, val in enumerate(sorted(populations[-1], key=lambda x: x[0], reverse=False)):
        #     print(val)
        # Population sorted by Test ID, not that at this point matters too much...
        return sorted(populations[-1], key=lambda x: x[0], reverse=False)

def main():
    # Parsing CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-log', help='Input Log File')

    args = parser.parse_args()

    # State: Gen, Evaluate, Evolve
    if args.input_log is None:
        exit(1)

    la = LogAnalyzer()
    la.process_log(args.input_log)


if __name__ == "__main__":
    main()
