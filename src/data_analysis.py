#
# Extract data from input file, build CarState objects and compute stats about OBEs
import glob
import os.path
import json
import math
import csv
import argparse
import tempfile


# from itertools import zip

from shapely.geometry import Point

from asfault.tests import CarState
from asfault.tests import RoadTest

# TODO Check that this is actually the case !!
global DEFAULT_LANE_WIDTH
global TICK_DURATION


def pairwise(iterable):
    """
    Return a sliding window of size n over the input iterable, as per:
    https://stackoverflow.com/questions/38151445/iterate-over-n-successive-elements-of-list-with-overlapping

    s -> (s0,s1), (s2,s3), (s4, s5), ...

    :param iterable:
    :return:
    """
    # a = iter(iterable)
    return list(zip(iterable, iterable[1:])) #zip(a, a)


def computeAreaOfTriangle(a, b, c):
    """
    Compute the area of triangle given coordinates of its vertices as per:
    https://www.mathopenref.com/coordtrianglearea.html

    :param a:
    :param b:
    :param c:
    :return:
    """
    return math.fabs((a.x*(b.y-c.y) + b.x*(c.y-a.y)+c.x*(a.y-b.y))/2)


def computeIntensity( obePoints ):
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
        A1 = computeAreaOfTriangle(obe1[0], obe1[1], obe2[1])
        A2 = computeAreaOfTriangle(obe1[0], obe2[0], obe2[1])
        # print("Adding A1 ", A1)
        # print("Adding A2 ", A2)
        obeIntensity += A1 + A2
        count += 1

    # print( "final count", count )
    return obeIntensity

def outputAnObeAsCSV(outpuCSV, argList):
    # print("Write:", ','.join([str(x) for x in argList]))

    with open(outpuCSV, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(argList)

    csvFile.close()

def createOutputCSV(outpuCSV, argList):
    # print("Create file :", outpuCSV)
    # print("Create Header", ','.join([str(x) for x in argList]))

    with open(outpuCSV, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(argList)

    csvFile.close()

def processTestFile( inputJSON, outputCSV ):
    """
    Analyze a JSON test output file from AsFault and generate a line (a CSV) of its stats which include
    testID, OBEID, OBE Intensity Approximation

    Note that we do not include the entering and exiting OBE points in the approximation and we do not
    consider duration in time of an OBE because state observations are collected every 10 frames, and frame
    rate is not constant.

    :param inputJSON:
    :return:
    """

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

    for idx, state_dict in enumerate(states):
        # Parse input
        carstate = CarState.from_dict(roadTest, state_dict)

        # Distance from Center of the lane
        distance = carstate.get_centre_distance();
        if distance > DEFAULT_LANE_WIDTH / 2.0:
            if not isOBE:
                # print("OBE started at", idx)
                obeStartTick = idx

            # print("Keep going with OBE at", idx, "With distance", distance)
            # Keep counting for current OBE
            obeLength += 1

            obeDistance.append(distance - DEFAULT_LANE_WIDTH / 2.0)
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
            intensity = (distance - DEFAULT_LANE_WIDTH / 2.0)
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
                # Compute Duration in ticks and time
                totalOBEDuration = obeLength * TICK_DURATION / 1000;
                # Compute Intensity using the 2-triangle approximation
                totalOBEIntensity = computeIntensity(obePoints)
                # AVG Distance
                avgDistance = sum(obeDistance) / len(obeDistance)
                # Output to console
                outputAnObeAsCSV(outputCSV, [roadTest.test_id, obeCount, obeStartTick, obeEndTick, obeLength,
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
        # Compute Duration in ticks and time
        totalOBEDuration = obeLength * TICK_DURATION / 1000;
        # Compute Intensity using the 2-triangle approximation
        totalOBEIntensity = computeIntensity(obePoints)
        # AVG Distance
        avgDistance = sum(obeDistance) / len(obeDistance)
        # Output to console
        outputAnObeAsCSV(outputCSV, [roadTest.test_id, obeCount, obeStartTick, obeEndTick, obeLength, avgDistance,
                         totalOBEIntensity])
    # Double check that obeCount matches the reported count
    if obeCount != dictdump["execution"]["oobs"]:
        print("OBE count does not match for test ", dictdump["test_id"], ". (file ", inputJSON, ")",
              "Found ", obeCount, " OBEs while file reports", dictdump["execution"]["oobs"])



# General setup
DEFAULT_LANE_WIDTH =4.0 # meter
# THE FOLLOWING IS WRONG
TICK_DURATION = 250 # msec

# Parsing CLI
parser = argparse.ArgumentParser()
parser.add_argument('--output-csv', help='Output CSV')
parser.add_argument('--input-folder', help='Input folder where to find - non recursively - the test results')

args = parser.parse_args()

if args.output_csv is None:
    # The following create a file nevertheless, but it is empty
    outputCSV = tempfile.mkstemp(suffix='.csv')[1]
    print("Output CSV Not provided use temp file", outputCSV, "instead")
    createOutputCSV(outputCSV, ["testID", "obeID", "obeStartTick" , "obeEndTick", "obeLength", "avgDistance",
                         "totalOBEIntensity"])
else:
    outputCSV = args.output_csv
    print("Use output CSV ", outputCSV)

if os.path.isfile(outputCSV):
    print("File ", outputCSV, "exists. Append to it")
else:
    createOutputCSV(outputCSV, ["testID", "obeID", "obeStartTick", "obeEndTick", "obeLength", "avgDistance",
                         "totalOBEIntensity"])

# Process all the files in the input-folder
for inputJSON in glob.glob('/'.join([args.input_folder, 'test_*.json'])):
    processTestFile(inputJSON , outputCSV)

