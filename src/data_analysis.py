#
# Extract data from input file, build CarState objects and compute stats about OBEs
import glob
import os.path
import json
import math
import csv
import argparse
import tempfile

from shapely.geometry import Point

from asfault.tests import CarState, RoadTest, TestExecution
from asfault.app import ensure_environment, DEFAULT_ENV
from asfault.tests import RoadTest
from asfault import config as c





class DataAnalyzer:

    roadTest = None
    testExecution = None
    HEADER = ["testID", "obe_id", "obe_start", "obe_end", "obe_duration", "avg_distance", "cumul_distance"]

    def __get_empty_obe(self):
        return [self.roadTest.test_id, 0, 0, 0, 0, 0, 0]

    def __init__(self):
        ensure_environment(DEFAULT_ENV)



    def outputAnObeAsCSV(self, outpuCSV, argList):
        # print("Write:", ','.join([str(x) for x in argList]))

        with open(outpuCSV, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(argList)

        csvFile.close()

    def createOutputCSV(self, outpuCSV, argList):
        # print("Create file :", outpuCSV)
        # print("Create Header", ','.join([str(x) for x in argList]))

        with open(outpuCSV, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(argList)

        csvFile.close()

    def get_obes(self, inputJSON):
        with open(inputJSON) as handle:
            dictdump = json.loads(handle.read())
        handle.close()

        # Parse the JSON to RoadTest
        self.roadTest= RoadTest.from_dict(dictdump)
        # Execution counts the OBE
        self.testExecution = TestExecution.from_dict(self.roadTest, dictdump["execution"])
        #
        return self.testExecution.get_obes()


    def processTestFile(self, inputJSON, outputCSV ):
        """
        Analyze a JSON test output file from AsFault and generate a line (a CSV) of its stats which include
        testID, OBEID, OBE Intensity Approximation

        Note that we do not include the entering and exiting OBE points in the approximation and we do not
        consider duration in time of an OBE because state observations are collected every 10 frames, and frame
        rate is not constant.

        :param inputJSON:
        :return:
        """

        if outputCSV is None:
            # The following create a file nevertheless, but it is empty
            outputCSV = tempfile.mkstemp(suffix='.csv')[1]
            print("Output CSV Not provided use temp file", outputCSV, "instead")
            self.createOutputCSV(outputCSV, self.HEADER)
        else:
            if os.path.isfile(outputCSV):
                print("File ", outputCSV, "exists. Append to it")
            else:
                self.createOutputCSV(outputCSV, self.HEADER)

        print("Process test file ", os.path.abspath(inputJSON), "and store data into", os.path.abspath(outputCSV))

        with open(inputJSON) as handle:
            dictdump = json.loads(handle.read())

        # Parse the JSON to RoadTest
        self.roadTest= RoadTest.from_dict(dictdump)
        # Execution counts the OBE
        self.testExecution = TestExecution.from_dict(self.roadTest, dictdump["execution"])

        obes = self.testExecution.get_obes()
        if len(obes) == 0:
            # We always include a line, corresponding to O OBE, for each test so we can track also tests which do not have any
            # Since everything default to 0 this shall not count when computing cumulative values
            self.outputAnObeAsCSV(outputCSV, self.__get_empty_obe())
        else:
            # Process the OBEs
            for idx, obe in enumerate(obes):
                # print("Processing OBE", idx, obe.get_start(), obe.get_end(), obe.get_duration(),
                #          obe.get_cumulative_distance(), obe.get_average_distance())

                # Output to console
                self.outputAnObeAsCSV(outputCSV, [self.roadTest.test_id, idx, obe.get_start(), obe.get_end(), obe.get_duration(),
                         obe.get_average_distance(), obe.get_cumulative_distance()])

        if len(obes) != dictdump["execution"]["oobs"]:
            print("OBE count does not match for test ", dictdump["test_id"], ". (file ", inputJSON, ")",
                  "\n\tFound ", len(obes), " OBEs while file reports", dictdump["execution"]["oobs"])


def main():
    # Parsing CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-csv', help='Output CSV')
    parser.add_argument('--input-folder', help='Input folder where to find - non recursively - the test results')
    parser.add_argument('--input-json', help='Input JSON')

    args = parser.parse_args()
    da = DataAnalyzer()

    # Process all the files in the input-folder
    if args.input_json is not None:
        da.processTestFile(args.input_json , args.output_csv)
    else:
        for inputJSON in glob.glob('/'.join([args.input_folder, 'test_*.json'])):
            da.processTestFile(inputJSON,args.output_csv)

if __name__ == "__main__":
    main()
