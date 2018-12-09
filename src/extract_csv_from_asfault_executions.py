import sys
import argparse
import glob
import re
import os.path

from log_analysis import LogAnalyzer
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


def main():
    # Cannot use execution bucket as unique ID for the experiment since this is reset day by day
    # Also the global counter might not be perfect !
    global_experiment_id = 0

    # Parse the CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-folder', help='Input Log File')

    args = parser.parse_args()
    # Ensure the trailing / is there
    root_folder = os.path.join(args.root_folder, '')
    print("ROOT FOLDER", root_folder)

    # Process all the execution.log files found under root_folder
    for experiment_log_file in glob.iglob('/'.join([root_folder, '**', 'experiment.log']), recursive=True):
        global_experiment_id += 1
        try:
            print("Found", experiment_log_file )

            ## TEMPORARY FILTER
            if not asfault_regex.match(experiment_log_file):
                print("Skip (no asfault)")
                continue

            if not small_map_regex.match(experiment_log_file):
                print("Skip (no small)")
                continue

            if not multi_regex.match(experiment_log_file):
                print("Skip (no multi)")
                continue

            # CAP THE SIMULATION
            la = LogAnalyzer(GENERATION_LIMIT=40)
            da = DataAnalyzer()

            final_test_suite = la.process_log(experiment_log_file )

            for test in final_test_suite:
                testID = test[0];
                # print("Processing Test ", testID)
                inputJSON= os.path.join(os.path.split(os.path.abspath(experiment_log_file))[0],
                                        'output', 'execs', ''.join(['test_', testID.zfill(4), '.json']));

                # Check if exists otherwise look under /final
                if not os.path.isfile(inputJSON):
                    inputJSON = os.path.join(os.path.split(os.path.abspath(experiment_log_file))[0],
                                             'output', 'final', ''.join(['test_', testID.zfill(4), '.json']));

                # Go two directories above the log file and get the folder name.
                parent = os.path.split(os.path.abspath(experiment_log_file))[0]
                gran_parent = os.path.split(os.path.abspath(parent))[0]
                # executionbucket=os.path.split(os.path.abspath(gran_parent))[1]

                # outputCSV = None
                cardinality = None
                generator = None
                # mapSize = None

                if random_regex.match( experiment_log_file ):
                    generator = "random"
                elif asfault_regex.match( experiment_log_file ) :
                    generator = "asfault"
                else:
                    print("ERROR: Unknown Generator for", experiment_log_file, " Skipping it!")
                    continue

                if single_regex.match( str(gran_parent), re.IGNORECASE):
                        cardinality = "single"
                elif multi_regex.match( str(gran_parent), re.IGNORECASE):
                        cardinality = "multi"
                else:
                    print("ERROR: Unknown Cardinality for", gran_parent, " Skipping it!")
                    continue


                if tiny_map_regex.match(experiment_log_file):
                    mapSize = "tiny"
                elif small_map_regex.match( experiment_log_file):
                    mapSize = "small"
                elif large_map_regex.match( experiment_log_file ):
                    mapSize = "large"
                else:
                    print("ERROR: Unknown map size for", experiment_log_file, " Skipping it!")
                    continue

                outputCSV = '.'.join([ '_'.join([generator,cardinality,mapSize, str(global_experiment_id)]), 'csv'])

                # This might fail because FILEs are somehow missing so we need to trap the error
                try:
                    da.processTestFile(inputJSON, outputCSV)
                except:
                    print("There was an error processing TEST", inputJSON)
                    # This invalidate the entire experimentss
                    raise
        except Exception as e:
            print("Experiment RUN ", experiment_log_file, "is INVALID !", e)

if __name__ == "__main__":
    main()

