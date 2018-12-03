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
random_regex = re.compile(r".*random.*$")
# TODO Those might require some adjustment
single_regex = re.compile(r".*single.*$")
multi_regex = re.compile(r".*multi.*$")

tiny_map_regex = re.compile(r".*_0500_.*$")
small_map_regex = re.compile(r".*_1000_.*$")
large_map_regex = re.compile(r".*_2000_.*$")




def main():
    # Parse the CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-folder', help='Input Log File')

    args = parser.parse_args()
    # Ensure the trailing / is there
    root_folder = os.path.join(args.root_folder, '')
    print("ROOT FOLDER", root_folder)

    # Process all the execution.log files found under root_folder
    for experiment_log_file in glob.iglob('/'.join([root_folder, '**', 'experiment.log']), recursive=True):
        print("Found", experiment_log_file )
        la = LogAnalyzer()
        final_test_suite = la.process_log(experiment_log_file )
        for test in final_test_suite:
            testID = test[0];
            print("Processing Test ", testID)
            da = DataAnalyzer()
            inputJSON= os.path.join(os.path.split(os.path.abspath(experiment_log_file))[0],
                                  'output', 'execs', ''.join(['test_', testID.zfill(4), '.json']));

            # Check if exists otherwise look under /final
            if not os.path.isfile(inputJSON):
                inputJSON = os.path.join(os.path.split(os.path.abspath(experiment_log_file))[0],
                                         'output', 'final', ''.join(['test_', testID.zfill(4), '.json']));

            # Go two directories above the log file and get the folder name.
            parent = os.path.split(os.path.abspath(experiment_log_file))[0]
            gran_parent = os.path.split(os.path.abspath(parent))[0]
            executionbucket=os.path.split(os.path.abspath(gran_parent))[1]

            outputCSV = None
            if random_regex.match( experiment_log_file ):
                outputCSV = '.'.join(['random_single_large', executionbucket,'csv'])
            elif asfault_regex.match( experiment_log_file ) :
                cardinality = "multi"
                mapSize = "large"

                if single_regex.match( experiment_log_file, re.IGNORECASE):
                    cardinality = "single"

                if tiny_map_regex.match(experiment_log_file):
                    mapSize = "tiny"
                elif small_map_regex.match( experiment_log_file):
                    mapSize = "small"

                outputCSV = '.'.join([ '_'.join(['asfault',cardinality,mapSize]), executionbucket, 'csv'])


            da.processTestFile(inputJSON, outputCSV)




if __name__ == "__main__":
    main()

