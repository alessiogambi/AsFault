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

    # This identify a test execution
    test_execution_regex = re.compile(r".*Executing Test#(\d+) .*$")

    # This mark the beginning of the evolution step
    # execution_regex = re.compile(f".*Evaluating test suite after evolution step..*$")

    # This mark the end of the (previous) evolution step
    evolution_regex = re.compile(r".*Test evolution step: (\d+)")

    POPULATION_SIZE = 25
    GENERATION_LIMIT = -1

    evolutionStep = 0
    test_executions_folder = None
    test_final_folder = None

    def __init__(self, GENERATION_LIMIT=-1):
        ensure_environment(DEFAULT_ENV)
        self.GENERATION_LIMIT = GENERATION_LIMIT

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

    def process_log(self, input_log):
        self.test_executions_folder = '/'.join([os.path.dirname(input_log), 'output/execs'])
        self.test_final_folder = '/'.join([os.path.dirname(input_log), 'output/final'])

        # Current population
        population = []

        # Populations in each step of the evolution
        populations = []

        # Open input file in 'read' mode
        with open(input_log, "r") as in_file:
            for line in in_file:

                test_execution_match = self.test_execution_regex.match(line)
                evolution_match = self.evolution_regex.match(line)

                if evolution_match is not None:
                    # print("Evolution Step Completed")

                    # Check how many tests are in the population
                    if len(population) < 25:
                        ## TODO Count how many tests are rejected at every cycle !
                        # print("***** Missing ", (25 - len(population)), " tests.")
                        # Take the last element in the populations list, which is the first from right, i.e., -1
                        sorted_population = sorted(populations[-1], key=lambda x: x[1], reverse=True)
                        # print("Sorted population: ", sorted_population);
                        for idx, val in enumerate(sorted_population):
                            # print("Adding ", idx, val, "to population")
                            population.append(val)
                            if len(population) == 25:
                                break
                    # At this point we have the population at evolution step and we store it
                    populations.append(population)
                    # print("Simulated evolution step", len(populations))

                    # Reset counters and states
                    population = []

                elif test_execution_match is not None:
                    # print("Test Execution Found")
                    # Extract test ID
                    testID = test_execution_match.group(1);
                    fitness = self.getFitnessForTest(testID)
                    # print("Adding ", testID, "to population with fitness", fitness)
                    population.append((testID, fitness))

                # We cap the generation since many test files are missing !
                if self.GENERATION_LIMIT > -1 and len(populations) >= self.GENERATION_LIMIT:
                    print("Reached GENERATION_LIMIT", self.GENERATION_LIMIT)
                    return sorted(populations[-1], key=lambda x: x[0], reverse=False)

        print("Final GENERATION count", len(populations))
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
