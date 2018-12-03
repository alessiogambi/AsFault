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

from asfault.tests import CarState, RoadTest
from asfault.app import ensure_environment, DEFAULT_ENV
from asfault.tests import RoadTest
from asfault import config as c


def pairwise(iterable):
    """
    Return a sliding window of size n over the input iterable, as per:
    https://stackoverflow.com/questions/38151445/iterate-over-n-successive-elements-of-list-with-overlapping

    s -> (s0,s1), (s2,s3), (s4, s5), ...

    :param iterable:
    :return:
    """
    # a = iter(iterable)
    return list(zip(iterable, iterable[1:]))  # zip(a, a)



class DataAnalyzer:

    def __init__(self):
        ensure_environment(DEFAULT_ENV)

    def computeAreaOfTriangle(self, a, b, c):
        """
        Compute the area of triangle given coordinates of its vertices as per:
        https://www.mathopenref.com/coordtrianglearea.html

        :param a:
        :param b:
        :param c:
        :return:
        """
        return math.fabs((a.x*(b.y-c.y) + b.x*(c.y-a.y)+c.x*(a.y-b.y))/2)


    def computeIntensity( self, obePoints ):
        """
        The intensity of an OBE is defined as the area outside the lane it creates. We approximate this by means of two
        triangles defines by four points: the measured state and the lane intersect of two consecutive out-of-lane
        measurements.

        :param obePoints:
        :return:
        """
        # print(len(obePoints), " OBEs")
        # print("Expected count", (len(obePoints) - 1))
        count = 0
        obeIntensity = 0
        # Note the use of pairwise
        for obe1, obe2 in pairwise(obePoints):
            # print(obe1[0].x, "->", obe2[0].x)
            # Computing the v12 between obes. OBE[0] is the measured state/point while OBE[1] is the lane intersect point
            A1 = self.computeAreaOfTriangle(obe1[0], obe1[1], obe2[1])
            A2 = self.computeAreaOfTriangle(obe1[0], obe2[0], obe2[1])
            # print("Adding A1 ", A1)
            # print("Adding A2 ", A2)
            obeIntensity += A1 + A2
            count += 1

        # print( "final count", count )
        return obeIntensity

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

        print("Process test file ", os.path.abspath(inputJSON), "and store data into", os.path.abspath(outputCSV))

        if outputCSV is None:
            # The following create a file nevertheless, but it is empty
            outputCSV = tempfile.mkstemp(suffix='.csv')[1]
            print("Output CSV Not provided use temp file", outputCSV, "instead")
            self.createOutputCSV(outputCSV, ["testID", "obeID", "obeStartTick", "obeEndTick", "obeLength", "avgDistance",
                                           "totalOBEIntensity"])
        else:
            if os.path.isfile(outputCSV):
                print("File ", outputCSV, "exists. Append to it")
            else:
                self.createOutputCSV(outputCSV, ["testID", "obeID", "obeStartTick", "obeEndTick", "obeLength", "avgDistance",
                                           "totalOBEIntensity"])

        with open(inputJSON) as handle:
            dictdump = json.loads(handle.read())

        # Parse the JSON to RoadTest
        roadTest= RoadTest.from_dict(dictdump)

        # Extract the list of logged states from the execution
        states = dictdump["execution"]["states"]

        isOBE = False
        obeStartTick = -1
        obeEndTick = -1
        obeCount = 0
        obeLength = 0
        obeDistance = []
        obePoints = []

        # We always include a line, corresponding to O OBE, for each test so we can track also tests which do not have any
        # Since everything default to 0 this shall not count when computing cumulative values
        self.outputAnObeAsCSV(outputCSV, [roadTest.test_id, 0, 0, 0, 0, 0, 0])

        for idx, state_dict in enumerate(states):
            # Parse input
            carstate = CarState.from_dict(roadTest, state_dict)

            # Distance from Center of the lane
            distance = carstate.get_centre_distance();
            if distance > c.ev.lane_width / 2.0:
                if not isOBE:
                    # print("OBE started at", idx)
                    obeStartTick = idx

                # print("Keep going with OBE at", idx, "With distance", distance)
                # Keep counting for current OBE
                obeLength += 1

                obeDistance.append(distance - c.ev.lane_width / 2.0)
                # Get the Path Project... Whatever this is
                projected = carstate.get_path_projection()
                # print("projected ", projected )
                # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
                measured = Point(carstate.pos_x, carstate.pos_y)
                # print("measured", measured)
                # Compute the norm v / || v ||
                distV = math.fabs(projected.distance(measured))
                u = Point( ( projected.x - measured.x) / distV, (projected.y - measured.y) / distV );
                # U is a unitary vector which points to projected from measured
                # now we move distance -
                # print("u", u)
                ## Double check u must be 1 - (DEFATUL_LANE_WIDTH / 2.0) away from measured and we shall get the
                # Boundary of the road which is the closest to measured
                intensity = (distance - c.ev.lane_width / 2.0)
                laneIntersect = Point( measured.x  + u.x * intensity , measured.y + u.y * intensity)
                # du = math.sqrt(u.x*u.x + u.y*u.y)
                # print( "DU ", du)
                # print("Lane Intersect ", laneIntersect)
                # Accumulate the Points to compute the Intensity/Area
                obePoints.append((measured, laneIntersect))
                # else:
                # print("Got a new OBE at" ,idx, "With distance", distance)
                # Mark the beginning of a new OBE
                #     obeLength += 1
                #     obeDistance.append(distance)
                isOBE = True;
            # Car is in lane
            else:
                if isOBE:
                    # This mark the end of the OBE. Print a CSV line
                    obeCount += 1
                    # Compute OBE stats
                    obeEndTick = idx
                    # Compute Intensity using the 2-triangle approximation
                    totalOBEIntensity = self.computeIntensity(obePoints)
                    # AVG Distance
                    avgDistance = sum(obeDistance) / len(obeDistance)
                    # Output to console
                    self.outputAnObeAsCSV(outputCSV, [roadTest.test_id, obeCount, obeStartTick, obeEndTick, obeLength,
                                     avgDistance, totalOBEIntensity])
                    # RESET STATE
                    # isOBE = False
                    obeStartTick = -1
                    obeEndTick = -1
                    obeLength = 0
                    obeDistance = []
                    obePoints = []
                # else:
                    # Do nothing
                isOBE = False;

        # At the end if isOBE is still on we report one last OBE
        if isOBE:
            # This mark the end of the OBE. Print a CSV line
            obeCount += 1
            # Compute OBE stats
            obeEndTick = idx
            # Compute Intensity using the 2-triangle approximation
            totalOBEIntensity = self.computeIntensity(obePoints)
            # AVG Distance
            avgDistance = sum(obeDistance) / len(obeDistance)
            # Output to console
            self.outputAnObeAsCSV(outputCSV, [roadTest.test_id, obeCount, obeStartTick, obeEndTick, obeLength, avgDistance,
                             totalOBEIntensity])
        # Double check that obeCount matches the reported count
        if obeCount != dictdump["execution"]["oobs"]:
            print("OBE count does not match for test ", dictdump["test_id"], ". (file ", inputJSON, ")",
                  "Found ", obeCount, " OBEs while file reports", dictdump["execution"]["oobs"])



def main():
    # Parsing CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-csv', help='Output CSV')
    parser.add_argument('--input-folder', help='Input folder where to find - non recursively - the test results')

    args = parser.parse_args()
    da = DataAnalyzer()

    # Process all the files in the input-folder
    for inputJSON in glob.glob('/'.join([args.input_folder, 'test_*.json'])):
        da.processTestFile(inputJSON,args.output_csv)

if __name__ == "__main__":
    main()
