import hashlib
import argparse
import glob
import re
import os.path
import csv

from log_analysis import LogAnalyzer, PopulationStats
from data_analysis import DataAnalyzer

# This script assumes the following folder organization:
# <ROOT_FOLDER>/<timestamp>/<executionbucket>/<condition>
# ROOT_FOLDER can be Single or Multi, timestamp 2018-09-06T08-02-03, executionbucket a 3 digit int (000, 001), and condition
# can be <???>_lanedist_<???>_<???> or <???>random<???>

# The input is <home>
# The output is one or multiple files named as
#   random_single_large.<executionbucket>.csv
# 	asfault_single_large.<executionbucket>.csv
# 	asfault_multi_large.<executionbucket>.csv
# 	asfault_multi_small.<executionbucket>.csv

# Regex used to match relevant loglines (in this case, a specific IP address)
from setuptools.command.install import install
# /Users/gambi/Dropbox/MarcMuller/Exps/Multi/2018-09-06T08-02-03/000/000_lanedist_0500_0075/experiment.log
# ROOT_FOLDER /Users/gambi/Dropbox/MarcMuller/Exps/Multi/
# EXPERIMENT_LOCAL_FOLDER /2018-09-06T08-02-03/000/000_lanedist_0500_0075/experiment.log
# EXPERIMENT_LOCAL_FOLDER /2018-09-06T08-02-03/000/random/experiment.log

asfault_regex = re.compile(r".*_lanedist_.*$")
random_regex = re.compile(r".*_random_.*$")
# TODO Those might require some adjustment
single_regex = re.compile(r".*single.*$", re.IGNORECASE)
multi_regex = re.compile(r".*multi.*$", re.IGNORECASE)

tiny_map_regex = re.compile(r".*_0500_.*$")
small_map_regex = re.compile(r".*_1000_.*$")
large_map_regex = re.compile(r".*_2000_.*$")

def get_hash(file):
    """
        We use hash to uniquely identify experiments so we can pause and resume computations
    """
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(file, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()


def get_input_json_for_test(experiment_log_file, testID):

    # Check if output folder is where we expect
    output_folder = os.path.join(os.path.split(os.path.abspath(experiment_log_file))[0], 'output')

    if not os.path.exists(output_folder):
        output_folder = os.path.join(os.path.split(os.path.abspath(experiment_log_file))[0], '.asfaultenv', 'output')

    inputJSON = os.path.join(os.path.split(os.path.abspath(experiment_log_file))[0],
                             output_folder, 'execs', ''.join(['test_', testID.zfill(4), '.json']));

    # Check if the file exists under 'execs' folder, otherwise look under 'final' folder
    if not os.path.isfile(inputJSON):
        inputJSON = os.path.join(os.path.split(os.path.abspath(experiment_log_file))[0],
                                 output_folder, 'final', ''.join(['test_', testID.zfill(4), '.json']));

    return inputJSON


def do_timing_analysis(output_file, experiment_log_file, population_size):
    log_analyzer = LogAnalyzer(GENERATION_LIMIT=15, POPULATION_SIZE=population_size)
    populations = log_analyzer.process_log(experiment_log_file)

    print(">> Running Timing/Generation Analysis")
    print(">> Output to", output_file)

    with open(output_file, 'w') as csvFile:
        writer = csv.writer(csvFile)
        # Write Header
        writer.writerow(["evolution_step",
                         "evolved_individuals",
                         "padded_individuals",
                         "invalid_tests",
                         "filtered_tests",
                         "generation_time",
                         "execution_time"])
        for idx, population in enumerate(populations):
            print(population)
            writer.writerow([idx,
                             len(population.get_evolved_individuals()),
                             len(population.get_padded_individuals()),
                             population.get_invalid_tests(),
                             population.get_filtered_tests(),
                             population.get_test_generation_time(),
                             population.get_test_execution_time()])
        csvFile.close()


def do_fitness_obe_analysis(output_file, experiment_log_file, population_size):
    log_analyzer = LogAnalyzer(GENERATION_LIMIT=15, POPULATION_SIZE=population_size)
    populations = log_analyzer.process_log(experiment_log_file)

    print(">> Running fitness obe analysis")
    print(">> Output to", output_file)
    with open(output_file, 'w') as csvFile:
        try:
            writer = csv.writer(csvFile)
            # Write Header
            writer.writerow(["evolution_step",
                             "cumulative_obe",
                             "cumulative_fitness"])

            # For each population get cumulative OBE count and fitness value
            for idx, population in enumerate(populations):
                # print("Processing population", idx)
                cumulative_fitness_value = 0
                cumulative_OBE = 0
                # Accumulate values
                for test in population.get_individuals():
                    testID = test[0]
                    fitness = log_analyzer.getFitnessForTest(testID)
                    # Compute OBEs
                    data_analyzer = DataAnalyzer()
                    input_json = get_input_json_for_test(experiment_log_file, testID)
                    # This count the obe from the JSON file but it does NOT recompute it !!?!
                    obe_count = len(data_analyzer.get_obes(input_json))

                    cumulative_fitness_value += fitness
                    cumulative_OBE += obe_count

                    # print("Test ID", testID, obe_count, fitness)

                # At this point we can store to CSV FILE the entry for the population
                # print("Population", idx, cumulative_OBE, cumulative_fitness_value)
                writer.writerow([idx, cumulative_OBE, cumulative_fitness_value])
        except:
            print("There was an error processing TEST from", input_json)
            # This invalidate the entire experiment
            # Is this necessary of using the with keyword this is already taken care of?
            csvFile.close()
            raise

        csvFile.close()


def do_tests_analysis(output_file, experiment_log_file, population_size):
    log_analyzer = LogAnalyzer(GENERATION_LIMIT=15, POPULATION_SIZE=population_size)
    populations = log_analyzer.process_log(experiment_log_file)

    print(">> Running Tests Analysis")
    print(">> Output to", output_file)

    # Final test suite is the last PopulationStats
    final_test_suite = populations[-1]

    for test in final_test_suite.get_individuals():
        testID = test[0];
        print("Processing Test ", testID)
        data_analyzer = DataAnalyzer()

        input_json = get_input_json_for_test(experiment_log_file, testID)

        # This might fail because FILEs are somehow missing so we need to trap the error
        try:
            data_analyzer.processTestFile(input_json, output_file)
        except:
            print("There was an error processing TEST", input_json)
            # This invalidate the entire experimentss
            raise


## TODO Use the HASH of the file content to identify the experiments we have already processed
def main():
    # Cannot use execution bucket as unique ID for the experiment since this is reset day by day
    # Also the global counter might not be perfect !
    # global_experiment_id = 0
    # We cannot even use a global ID since we cannot ensure files a listed in the same order !

    # Parse the CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-folder', help='Input Log File')
    parser.add_argument('--tests-analysis', action='store_true')
    parser.add_argument('--timing-analysis', action='store_true')
    parser.add_argument('--fitness-obe-analysis', action='store_true')
    parser.add_argument('--only', help='Filter by generator random|asfault', action='store', nargs='?')
    parser.add_argument('--population-size', help='Size of the population', default='25')
    parser.add_argument('--output-folder', help='Folder to put the files', action='store', nargs='?')

    args = parser.parse_args()

    if args.root_folder is None:
        print("No Root Folder")
        exit(0)

    POPULATION_SIZE=int(args.population_size)
    print("Population SIZE =", POPULATION_SIZE)

    # Ensure the trailing / is there
    root_folder = os.path.join(args.root_folder, '')
    print("ROOT FOLDER", root_folder)

    output_folder = os.getcwd()
    if args.output_folder is not None:
        # Ensure the trailing / is there
        output_folder = os.path.join(args.output_folder, '')

    # Process all the execution.log files found under root_folder
    for experiment_log_file in glob.iglob('/'.join([root_folder, '**', 'experiment.log']), recursive=True):

        # Compute the hash from the
        log_hash = get_hash(experiment_log_file)

        try:
            print("Found", experiment_log_file, "with hash ", log_hash)

            # Extract metadata from file name
            # Go two directories above the log file and get the folder name.
            parent = os.path.split(os.path.abspath(experiment_log_file))[0]
            gran_parent = os.path.split(os.path.abspath(parent))[0]

            if random_regex.match(experiment_log_file):
                generator = "random"
            elif asfault_regex.match(experiment_log_file):
                generator = "asfault"
            else:
                print("ERROR: Unknown Generator for", experiment_log_file, " Skipping it!")
                continue

            if args.only is not None and generator != args.only:
                print("Skip", experiment_log_file, "as it does not match", args.only)
                continue

            if single_regex.match(str(gran_parent), re.IGNORECASE):
                cardinality = "single"
            elif multi_regex.match(str(gran_parent), re.IGNORECASE):
                cardinality = "multi"
            else:
                print("ERROR: Unknown Cardinality for", gran_parent, " Skipping it!")
                continue

            if tiny_map_regex.match(experiment_log_file):
                mapSize = "tiny"
            elif small_map_regex.match(experiment_log_file):
                mapSize = "small"
            elif large_map_regex.match(experiment_log_file):
                mapSize = "large"
            else:
                print("ERROR: Unknown map size for", experiment_log_file, " Skipping it!")
                continue

            # Setup the output files

            # Stats about the evolution and the populations
            populations_timing_stats_csv = '.'.join(
                ['_'.join([generator, cardinality, mapSize, str(log_hash), 'population', 'timing']), 'csv'])

            populations_timing_stats_csv = os.path.join(output_folder, populations_timing_stats_csv)

            populations_fitness_obe_stats_csv = '.'.join(
                ['_'.join([generator, cardinality, mapSize, str(log_hash), 'population', 'fitness', 'obe']), 'csv'])

            populations_fitness_obe_stats_csv = os.path.join(output_folder, populations_fitness_obe_stats_csv)

            # Stats about each test
            tests_analysis_csv = '.'.join(['_'.join([generator, cardinality, mapSize, str(log_hash)]), 'csv'])

            tests_analysis_csv = os.path.join( output_folder, tests_analysis_csv )

            # TODO Pre-compute the populations if more than one analysis is active

            if args.timing_analysis:
                if not os.path.exists(populations_timing_stats_csv):
                    do_timing_analysis(populations_timing_stats_csv, experiment_log_file, POPULATION_SIZE)
                else:
                    print(">> Skip Timing Analysis: output file exists", populations_timing_stats_csv)

            if args.fitness_obe_analysis:
                if not os.path.exists(populations_fitness_obe_stats_csv):
                    do_fitness_obe_analysis(populations_fitness_obe_stats_csv, experiment_log_file, POPULATION_SIZE)
                else:
                    print(">> Skip Fitness/OBE Analysis: output file exists", populations_fitness_obe_stats_csv)

            if args.tests_analysis:
                if not os.path.exists(tests_analysis_csv):
                    do_tests_analysis(tests_analysis_csv, experiment_log_file, POPULATION_SIZE)
                else:
                    print(">> Skip Tests Analysis: output file exists", tests_analysis_csv)

        except Exception as e:
            import traceback
            print("Experiment RUN ", experiment_log_file, "is INVALID !", e)
            traceback.print_tb(e.__traceback__)


if __name__ == "__main__":
    main()

