#
# Compute a path given the shape of the road and uses beamNGpy set Script
#
from beamngpy import BeamNGpy, Vehicle
from beamngpy.sensors import Electrics

from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt

import numpy as np
import sys
import logging as l

import math
from time import sleep


def log_exception(extype, value, trace):
    l.exception('Uncaught exception:', exc_info=(extype, value, trace))


def setup_logging():
    term_handler = l.StreamHandler()
    l.basicConfig(format='Driver AI: %(asctime)s %(levelname)-8s %(message)s',
                  level=l.INFO, handlers=[term_handler])
    sys.excepthook = log_exception


def pairs(lst):
    for i in range(1, len(lst)):
        yield lst[i - 1], lst[i]


def dot(a, b):
    return np.sum(a * b, axis=-1)


class RoadSegment:
    start = None
    end = None

    def __init__(self, length, max_speed, curvature):
        self.length = length
        self.max_speed = max_speed
        self.curvature = curvature

        # Store the road_segment objects from asfault
        self.asfault_road_segments = list()

class Road:
    _LIMIT = 100

    def __init__(self, road_segments):
        # Build the road by chaining the given RoadSegments
        self.road_segments = []
        # This is over x-axis only
        initial_position = 0


        for rs in road_segments:

            l.debug("Consider:", rs.length, ",", rs.max_speed)

            if len(self.road_segments )> 0 and self.road_segments[-1].max_speed == rs.max_speed:
                # We need to merge the two segments !
                # Compute start position is the initial position of the previous [-1]
                start = initial_position - self.road_segments[-1].length
                end = start + self.road_segments[-1].length + rs.length
                # NOTE: to have exactly the same speed the segments have the same curvature !
                curvature = self.road_segments[-1].curvature
                merged_length = end - start

                the_road_segment = RoadSegment(merged_length, rs.max_speed, curvature)
                # Include the rs from the previous segment and append the current rs
                the_road_segment.asfault_road_segments.extend(self.road_segments[-1].asfault_road_segments)
                the_road_segment.asfault_road_segments.extend(rs.asfault_road_segments)

                l.debug("Merging", self.road_segments[-1], the_road_segment)



                # Replace the road segment in the road
                self.road_segments[-1] = the_road_segment
            else:
                # Compute start and end
                start = initial_position
                end = start + rs.length
                curvature = rs.curvature

                the_road_segment = RoadSegment(rs.length, rs.max_speed, curvature)
                the_road_segment.asfault_road_segments.extend(rs.asfault_road_segments)

                # Store the road segment in the road
                self.road_segments.append(the_road_segment)

            # No matter what we update the last element in the list
            self.road_segments[-1].start = start
            self.road_segments[-1].end = end

            # Update loop variable
            initial_position = end
            # Keep track of the total length of the road
            self.total_length = end

    # Utility methods
    @staticmethod
    def _compute_acceleration_line(starting_point: Point, delta_v):
        # y1 = y0 + A * (x1 - x0)
        x0 = starting_point.x
        y0 = starting_point.y
        # Not sure why here is 50 ?
        x1 = x0 + 200
        y1 = y0 + delta_v * (x1 - x0)
        ending_point = Point(x1, y1)
        # Add a little epsilon here...
        x2 = x0 - 0.01
        y2 = y0 + delta_v * (x2 - x0)
        # Replace the original starting_point with the new one
        starting_point = Point(x2, y2)
        return LineString([starting_point, ending_point])

    @staticmethod
    def _compute_deceleration_line(starting_point: Point, delta_v):
        # y1 = y0 + A * (x1 - x0)
        x0 = starting_point.x
        y0 = starting_point.y
        x1 = x0 - 200
        y1 = y0 + delta_v * (x1 - x0)
        ending_point = Point(x1, y1)
        # Add a little epsilon here...
        x2 = x0 + 0.01
        y2 = y0 + delta_v * (x2 - x0)
        # Replace the original starting_point with the new one
        starting_point = Point(x2, y2)
        return LineString([starting_point, ending_point])


    # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    @staticmethod
    def _line(the_line_segment):
        """ A * x + B * y = C """

        assert len(list(zip(*the_line_segment.coords.xy))) == 2, "The line segment" + str(the_line_segment) + "is NOT defined by 2 points!"

        p1 = list(zip(*the_line_segment.coords.xy))[0]
        p2 = list(zip(*the_line_segment.coords.xy))[1]
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])

        return A, B, -C

    @staticmethod
    def _intersection(L1, L2):
        " Given two lines returns whether the lines intersect (x,y) or not "
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return Point(x, y)
        else:
            return False

    @staticmethod
    # Return the line, the distance, the new intersection point
    # Valid intersection points must be "after" (on the right than the input the_intersection_point
    def _find_closest_intersecting_line(the_line, the_intersection_point, the_other_lines):

        # TODO It might happen that in the same point more than two lines pass?
        # intersecting_lines = []
        closest_intersection_point_and_line = (Point(math.inf,0),  None)

        # This should work in the assumption the_line is defined only by two points, which should be the case for us !
        L1 = Road._line(the_line)

        for the_other_line in the_other_lines:
            L2 = Road._line(the_other_line)
            # Can be improved. Those are the first and last point that define that line
            p1 = Point(list(zip(*the_other_line.coords.xy))[0])
            p2 = Point(list(zip(*the_other_line.coords.xy))[1])

            R = Road._intersection(L1, L2)

            if R:
                # Since we are considering the lines that extend the segments for checking the intersection
                #   we must ensure that we are not intersecting OUTSIDE the segments

                if the_intersection_point.x < R.x <= closest_intersection_point_and_line[0].x and \
                        (p1.x <= R.x <= p2.x or p2.x <= R.x <= p1.x):
                    # l.debug("Intersection detected at", R)
                    # The only point in which speed can be 0 is the initial point
                    assert R.y > 0.0, "Speed is negative or zero"
                    # l.debug("It's a closest intersection point to", the_intersection_point.x)
                    closest_intersection_point_and_line = (R, the_other_line)
            else:
                # Overlapping or parallel
                l.debug("WARN, No single intersection point detected between", the_line, "=", L1, "and", the_other_line, "=", L2)

        # Either there's a line or None is returned
        if closest_intersection_point_and_line[1] is not None:
            return closest_intersection_point_and_line
        else:
            return None

    def _compute_acceleration_lines(self, max_acc):
        # Because of floating point computation, points might not be exactly on the line, which means that they will
        #   not interpreted as intersections, so we add a small epsilon to them
        right_points = [Point(0, 0)] + [Point(rs.end, rs.max_speed) for rs in self.road_segments]
        return [Road._compute_acceleration_line(rp, max_acc) for rp in right_points]

    def _compute_deceleration_lines(self, max_dec):
        left_points = [Point(rs.start, rs.max_speed) for rs in self.road_segments]
        return [Road._compute_deceleration_line(rp, max_dec) for rp in left_points]

    def _compute_max_achievable_speed(self, max_acc, max_dec, speed_limit):
        # Iteratively identify the intersection points among acceleration, deceleration and max_speed lines
        acceleration_lines = self._compute_acceleration_lines(max_acc)
        deceleration_lines = self._compute_deceleration_lines(max_dec)
        #
        max_speed_lines = [LineString([(rs.start-0.01, rs.max_speed), (rs.end+0.01, rs.max_speed)]) for rs in self.road_segments]

        # Get the x value of the furthest possible point
        last_point = max_speed_lines[-1].coords[-1][0]

        # Start from position 0 and speed 0:
        intersection_points = [Point(0, 0)]
        last_intersection_point = intersection_points[0]
        # Initially we have to accelerate
        current_line = acceleration_lines[0]
        # Keep track of the last interescting line to include the last point defining the LineString
        last_intersecting_line = current_line

        while True:
            # We look for intersections with segments of different type than current_line, because according to
            # the model we cannot accelerate more than what we are currently doing (or dec)

            lines_to_check = []
            if current_line not in acceleration_lines:
                # l.debug("Consider ACC")
                lines_to_check.extend(acceleration_lines)
            if current_line not in deceleration_lines:
                # l.debug("Consider DEC")
                lines_to_check.extend(deceleration_lines)
            if current_line not in max_speed_lines:
                # l.debug("Consider MAX SPEED")
                lines_to_check.extend(max_speed_lines)

            # Find the closest line that intersects the current_line (after the last intersection point).
            # Note that the last intersection points always belong to the current_line
            intersection = Road._find_closest_intersecting_line(current_line, last_intersection_point, lines_to_check)

            # If we did not find any new intersection point we stop
            # TODO Check that we actually reached the end of the road !
            if intersection is None:
                # TODO Replace this with: until there's lines to check, check them otherwise exit the loop...
                # l.debug("Cannot find additional intersections") #. Intersection Points", intersection_points)
                break

            # Remove current_line from the search set, reduce the search space
            if current_line in acceleration_lines:
                acceleration_lines.remove(current_line)
            if current_line in deceleration_lines:
                deceleration_lines.remove(current_line)
            if current_line in max_speed_lines:
                max_speed_lines.remove(current_line)

            # Store the new intersection point and update the loop variables
            intersection_points.append(intersection[0])
            last_intersection_point = intersection[0]

            # TODO: Remove all the acceleration/deceleration lines that are before the last intersection point
            current_line = intersection[1]
            last_intersecting_line = current_line

        # Add the very last intersection point corresponding to last_point on current_line
        # if last_intersection_point.x < last_point:
        # l.debug("Debug", "Add last point to conclude the LineString using the last intersecting line", last_intersecting_line )
        # Get the coefficients corresponding to this line segment

        L = Road._line(last_intersecting_line)
        vertical_line = LineString([(last_point, 0), (last_point, speed_limit + 10)])
        V = Road._line(vertical_line)

        R = Road._intersection(L, V)

        assert R, "No single intersection to identify final point"

        # l.debug("Final point is ", R)
        intersection_points.append(R)

        # last_intersection_point = current_line.intersection( vertical_line )
        # if last_intersection_point.type == 'Point':
        #     intersection_points.append( last_intersection_point )

        # The "max_achievable_speed" of the road is defined by the list of the
        # intersection points. This information can be used to compute the speed profile of the road given
        # a "discretization function"
        return LineString([p for p in intersection_points])

    def _get_curvature_between(self, begin, end):
        # Get the segments between begin and end
        segments = [ s for s in self.road_segments if s.end >= begin and s.start <= end ]
        # Order segments from longest to smallest
        # TODO Check this !!!
        segments.sort(key=lambda rs: rs.length, reverse=False)

        # Take the first. ASSUMPTION: Always one?
        return segments[0].curvature

    def compute_curvature_profile(self, curvature_bins, distance_step):
        # Compute length of segment
        # Compute curvature of segment as 1/radius:
        # - Use angles to decide wheter this is a left/right (positive/negative)
        # - curvature == 0 for straigth

        # Compute the curvatyre in each segment of the road, defined by distance_step
        # For pieces that belong to multiple segments use the majority/bigger one to decide, and then
        # lexycografic order
        start = 0
        stop = self.total_length
        number_of_segments = math.ceil(stop / distance_step)
        discretized_road = np.linspace(start, stop, num=number_of_segments, endpoint=True)

        observations = []
        for begin, end in zip(discretized_road[0:], discretized_road[1:]):
            observations.append(self._get_curvature_between(begin, end))

        # Hist contains always one element less than bins because you need to consecutive points to define a bin...
        # [a, b, c] -> bins(a-b, b-c)
        hist, bins = np.histogram(observations, bins=curvature_bins)

        total = sum(hist)

        # Normalize the profile to get the percentage
        hist = [bin / total for bin in hist]

        return hist

    def _get_speed_at_position(self, max_achievable_speed, x):
        # Find the segment in which x belongs

        # last_point_before_x
        p1 = [coord for coord in list(zip(*max_achievable_speed.coords.xy)) if coord[0] <= x][-1]

        # first_point_after_x. This might be empty/missing due to discretizations, so we return None
        after_x = [coord for coord in list(zip(*max_achievable_speed.coords.xy)) if coord[0] >= x]

        if len(after_x) == 0:
            l.debug("DEBUG, Cannot find any point after", x, "Return None")
            return None

        p2 = after_x[0]

        # Can there be a case in which only last_point_before_x is found ? What do we do then?
        if p1 == p2:
            return p1[1]
        else:
            # Interpolate since those are piecewise linear
            # Slope (y2-y1)/(x2-x1)
            slope = (p2[1]-p1[1])/(p2[0]-p1[0])
            # Y-intercept: y1 = slope * x1 + b -> b = y1 - slope*x1
            y_intercept = p1[1] - slope * p1[0]
            # Finally compute the value which correponds to x by pluggin it inside the formula:
            y = x * slope + y_intercept
            return y

    def _plot_debug(self, max_acc, max_dec, max_achievable_speed):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        rp = RoadPlotter(ax, 5)
        rp.plot_max_speed(self)
        rp.plot_acceleration_lines(self, max_acc)
        rp.plot_deceleration_lines(self, max_dec)
        # Keep using the same plot
        xv, yv = max_achievable_speed.coords.xy
        ax.plot(xv, yv, 'green')

    def compute_speed_profile(self, max_acc, max_dec, speed_limit, speed_bins, distance_step):
        # Return the "actual" speed that the car can reach given its acc/dec
        max_achievable_speed = self._compute_max_achievable_speed(max_acc, max_dec, speed_limit)

        # self._plot_debug(max_acc, max_dec, max_achievable_speed)

        # Compute the avg speed in each segment of the road, defined by distance_step
        start = 0
        stop = self.total_length
        number_of_segments = math.ceil(stop / distance_step)
        discretized_road = np.linspace(start, stop, num=number_of_segments, endpoint=True)

        # Compute the "avg" speed at each segment. The avg is computed by computing the mean of:
        # entry point, exit point, and "intersection" points inside max_achievable_speed
        average_speed = []

        for begin, end in zip(discretized_road[0:], discretized_road[1:]):
            speed_at_begin = self._get_speed_at_position(max_achievable_speed, begin)
            assert speed_at_begin is not None, "Speed at begin of segment cannot be None"
            assert speed_at_begin <= speed_limit + 0.01, "Speed at begin of segment is over speed limit"
            assert speed_at_begin >= 0.0, "Speed at begin of segment is negative " + str(speed_at_begin)

            speed_at_end = self._get_speed_at_position(max_achievable_speed, end)

            if speed_at_end <= 0 or speed_at_end > speed_limit + 0.01:
                self._plot_debug(max_acc, max_dec, max_achievable_speed)

            assert speed_at_end >= 0.0, "Speed at end of segment is negative " + str(speed_at_end)
            assert speed_at_end <= speed_limit + 0.01, "Speed at end of segment " + str(speed_at_end) + " is over speed limit" + str(speed_limit + 0.01)
            assert speed_at_end is not None, "Speed at end of segment cannot be None"

            # Get all the points at which the speed changes between "begin" and "end" on the max_achievable_speed line
            speeds = [s[1] for s in list(zip(*max_achievable_speed.coords.xy)) if begin < s[0] < end]

            observations = []
            observations.append(speed_at_begin)
            observations = observations + speeds
            observations.append(speed_at_end)

            # Compute the average speed in the segment
            average_speed.append(np.mean(observations))

        # Compute the histogram of the average speeds across all the pieces of road
        hist, bins = np.histogram(average_speed, bins=speed_bins)

        # DEBUG
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # pp = ProfilePlotter()
        # pp.plot_speed_profile(hist, bins, ax)

        total = sum(hist)

        # Normalize the profile to get the percentage
        hist = [bin / total for bin in hist]

        # DEBUG
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # pp = ProfilePlotter()
        # pp.plot_speed_profile(hist, bins, ax)

        return hist

    def get_position_at_distance(self, distance):
        # Find the segment
        cumulative_length = 0
        target_segment = None
        for segment in self.road_segments:
            cumulative_length += segment.length
            if cumulative_length >= distance:
                target_segment = segment
                break
        # Find the position inside the target_segment
        distance_to_end = cumulative_length - distance
        distance_from_start = target_segment.length - distance_to_end

        cumulative_length = 0
        for pair in pairs(target_segment.asfault_road_segments):
            cumulative_length += pair[0].distance(pair[1])
            if cumulative_length >= distance_from_start:
                # The point we look for is between the points that define this pair at distance D from the second point
                D = cumulative_length - distance_from_start
                the_point = LineString([pair[1], pair[0]]).interpolate(D)
                plt.plot(the_point.x, the_point.y, marker='s', color='black')
                return the_point
        return None

    def discretize(self, meters):
        """ Return the coordinates of the road at various positions"""
        distance = 0
        locations = list()
        # TODO probably we can add speed as well?
        while distance < self.total_length:
            locations.append((distance, self.get_position_at_distance(distance)))
            distance += meters

        return locations

class RoadPlotter:

    def __init__(self, ax, distance_step):
        self.ax = ax
        self.distance_step = distance_step

    def _plot_lineStrings(self, lineStrings, color):
        for lineString in lineStrings:
            x,y = lineString.xy
            self.ax.plot(x, y, color = color)

    def _draw_grid(self, road: Road):
        # Draw grid again
        # Show the grid that corresponds to the discretization
        self.ax.xaxis.grid(True)
        # Compute the avg speed in each segment of the road, defined by distance_step
        start = 0
        stop = road.total_length
        number_of_segments = math.ceil(stop / self.distance_step)
        discretized_road = np.linspace(start, stop, num=number_of_segments, endpoint=True)
        # Compute the avg speed in each segment of the road, defined by distance_step
        ticks = discretized_road
        self.ax.set_xticks(ticks)

    def plot_acceleration_lines(self, road: Road, max_acc:float):
        self._plot_lineStrings(road._compute_acceleration_lines(max_acc), 'red')

    def plot_deceleration_lines(self, road: Road, max_dec:float):
        self._plot_lineStrings(road._compute_deceleration_lines(max_dec), 'blue')

    def plot_max_speed(self, road: Road):
        for road_segment in road.road_segments:
            # Plot max speed
            x = [road_segment.start, road_segment.end]
            y = [road_segment.max_speed, road_segment.max_speed]
            self.ax.plot(x, y, color='#999999')
        # self._draw_grid(road)


class RoadProfiler:

    G = 9.81
    _DISTANCE_STEP = 5  # Meters

    def __init__(self, mu, speed_limit_meter_per_second, discretization_factor=10):
        """
        :param mu: Friction coefficient
        """
        self.FRICTION_COEFFICIENT = mu
        self.speed_limit_meter_per_second = speed_limit_meter_per_second
        # Decide how fine grained the control his. Smaller values results in better controls but higher computational costs
        self.discretization_factor = discretization_factor

    def _compute_max_speed(self, radius: float):
        max_speed = self.speed_limit_meter_per_second

        if radius == math.inf:
            return max_speed

        # Assumption: All turns are unbanked turns. Pivot_off should be the radius of the curve
        max_speed = math.sqrt(self.FRICTION_COEFFICIENT * self.G * radius)
        if (max_speed >= self.speed_limit_meter_per_second):
            max_speed = self.speed_limit_meter_per_second

        return max_speed

    def _compute_curvature(self, radius: float):
        """ NOTE: We do not distinguish left and right turns here..."""
        if radius == math.inf:
            return 0
        else:
            return 1.0 / radius

    def _compute_radius_turn(self, road_segment: list): # List(Point)
        if len(road_segment) == 2:
            # Straight segment
            return Point(math.inf, math.inf), math.inf

        # Use triangulation.
        p1 = Point(road_segment[0])
        x1 = p1.x
        y1 = p1.y

        p2 = Point(road_segment[-1])
        x2 = p2.x
        y2 = p2.y

        # This more or less is the middle point, not that should matters
        p3 = Point(road_segment[int(len(road_segment) / 2)])
        x3 = p3.x
        y3 = p3.y

        center_x = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (
                    y1 - y2)) / (2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2))
        center_y = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (
                    x2 - x1)) / (2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2))
        radius = math.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)

        return Point(center_x, center_y), radius

    def _build_road_object(self, driving_path: LineString):
        # This works under the assumption that consecutive points in the driving_path at the same distance
        # belong to the same segment.

        plt.figure(1)

        previous_distance = None
        segments = list()
        current_segment = list()

        segments.append(current_segment)
        color = 'green'
        for pair in pairs(list(driving_path.coords[:])):
            A = Point(pair[0][0], pair[0][1])
            B = Point(pair[1][0], pair[1][1])
            d = A.distance(B)

            d = round(d, 2)

            # This has been found empirically...
            if previous_distance is None or math.fabs(d - previous_distance) < 0.5:
                # Add the points to current_segment
                # TODO Probably we can remove duplicates
                if A not in current_segment:
                    current_segment.append(A)
                if B not in current_segment:
                    current_segment.append(B)
            else:
                current_segment = list()
                segments.append(current_segment)
                current_segment.append(A)
                current_segment.append(B)
                color = 'red' if color == 'green' else 'green'

            previous_distance = d
            # plt.plot([A.x, B.x], [A.y, B.y], marker="o", color=color)

        input_road_segments = list()
        for road_segment in segments:
            center_point, radius = self._compute_radius_turn(road_segment)

            if radius != math.inf:
                plt.plot(center_point.x, center_point.y, marker="*")

            curvature = self._compute_curvature(radius)
            max_speed = self._compute_max_speed(radius)

            # Approximate length as cumulative distance between the points that define the segment
            length = 0

            for pair in pairs(road_segment):
                length += pair[0].distance(pair[1])

            the_road_segment = RoadSegment(length, max_speed, curvature)
            # TODO Instead of the asfault road segments we pass the list of points defining this segment...
            the_road_segment.asfault_road_segments.extend(road_segment)

            input_road_segments.append(the_road_segment)

        return Road(input_road_segments)

    def compute_ai_script(self, driving_path: LineString, car_model: dict):
        # From the driving path, a geometry, generate the data structure to compute the speed profile
        input_road = self._build_road_object(driving_path)

        # Locations is a list of tuples (distance, Point) computed from driving path every 10 meters
        # with 10 meter for discretization the driver cuts sharp turns
        locations = input_road.discretize(self.discretization_factor)

        # Return the "actual" speed that the car can reach given its acc/dec
        max_achievable_speed = input_road._compute_max_achievable_speed(car_model['max_acc'], car_model['max_dec'],
                                                                        self.speed_limit_meter_per_second)

        # input_road._plot_debug(car_model['max_acc'], car_model['max_dec'], max_achievable_speed)

        # Speeds is similar to Locations but contains distance and speed data
        speeds = list()
        for distance, _ in locations:
            speeds.append((distance, input_road._get_speed_at_position(max_achievable_speed, distance)))


        location_and_speed = list()
        for location, speed in zip(locations, speeds):
            location_and_speed.append((location[0], location[1], speed[1]))


        # Now we compute the time using constant speed law
        timing = list()
        cumulative_time = 0
        for pair in pairs(location_and_speed):
            avg_speed = 0.5 * (pair[0][2] + pair[1][2])
            distance = pair[1][0] - pair[0][0]
            travel_time = distance / avg_speed
            cumulative_time += travel_time
            timing.append(cumulative_time)

        assert len(timing) == len(location_and_speed) -1

        script = list()
        # We do  not report the initial position t=0, d=0 and v=0
        for location_and_speed, target_time in zip(location_and_speed[1:], timing[:]):
            travel_distance = location_and_speed[0]
            target_position = location_and_speed[1]
            avg_speed = location_and_speed[2]
            node = {
                'travel_distance': travel_distance,
                'avg_speed': avg_speed,
                'x': target_position.x,
                'y': target_position.y,
                'z': 0.3,
                't': target_time,
            }
            script.append(node)

        return script


class Driver:
    vehicle = None
    bng = None
    labeler = None

    left_path = None
    right_path = None
    driving_path = None

    road_profiler = None

    car_model = None
    road_model = None

    def __init__(self, car_model: dict, road_model: dict, control_model: dict):
        self.car_model = car_model
        self.road_model = road_model
        self.control_model = control_model

        # Note: Speed limit must be m/s
        self.road_profiler = RoadProfiler(road_model['mu'], road_model['speed_limit']/ 3.6, control_model['discretization_factor'])

    def _compute_driving_path(self, car_state, road_name):
        road_geometry = self.bng.get_road_edges(road_name)

        left_edge_x = np.array([e['left'][0] for e in road_geometry])
        left_edge_y = np.array([e['left'][1] for e in road_geometry])

        right_edge_x = np.array([e['right'][0] for e in road_geometry])
        right_edge_y = np.array([e['right'][1] for e in road_geometry])

        road_edges = dict()
        road_edges['left_edge_x'] = left_edge_x
        road_edges['left_edge_y'] = left_edge_y
        road_edges['right_edge_x'] = right_edge_x
        road_edges['right_edge_y'] = right_edge_y

        self.right_edge = LineString(zip(road_edges['right_edge_x'][::-1], road_edges['right_edge_y'][::-1]))
        self.left_edge = LineString(zip(road_edges['left_edge_x'], road_edges['left_edge_y']))

        current_position = Point(car_state['pos'][0], car_state['pos'][1])

        from shapely.ops import nearest_points
        from shapely.affinity import rotate

        projection_point_on_right = nearest_points(self.right_edge, current_position)[0]
        projection_point_on_left = nearest_points(self.left_edge, current_position)[0]

        # If the car is closest to the left, then we need to switch the direction of the road...
        if current_position.distance(projection_point_on_right) > current_position.distance(projection_point_on_left):
            # Swap the axis and recompute the projection points
            l.debug("Reverse traffic direction")
            temp = self.right_edge
            self.right_edge = self.left_edge
            self.left_edge = temp
            del temp

            projection_point_on_right = nearest_points(self.right_edge, current_position)[0]
            projection_point_on_left = nearest_points(self.left_edge, current_position)[0]




        # Traffic direction is always 90-deg counter clockwise from right
        # Now rotate right point 90-deg counter clockwise from left and we obtain the traffic direction
        rotated_right = rotate(projection_point_on_right, 90.0, origin=projection_point_on_left)

        # Vector defining the direction of the road
        traffic_direction = np.array(
            [rotated_right.x - projection_point_on_left.x, rotated_right.y - projection_point_on_left.y])

        # Find the segment containing the projection of current location
        # Starting point on right edge


        start_point = None
        for pair in pairs(list(self.right_edge.coords[:])):
            segment = LineString([pair[0], pair[1]])
            # xs, ys = segment.coords.xy
            # plt.plot(xs, ys, color='green')
            if segment.distance(projection_point_on_right) < 1.8e-5:
                road_direction = np.array([pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]])
                if dot(traffic_direction, road_direction) < 0:
                    l.debug("Reverse order !")
                    self.right_edge = LineString([Point(p[0], p[1]) for p in self.right_edge.coords[::-1]])
                    start_point = Point(pair[0][0], pair[0][1])
                    break
                else:
                    l.debug("Original order !")
                    start_point = Point(pair[1][0], pair[1][1])
                    break

        assert start_point is not None

        # At this point compute the driving path of the car (x, y, t)
        self.driving_path = [current_position]
        # plt.plot(current_position.x, current_position.y, color='black', marker="x")
        # # This might not be robust we need to get somethign close by
        # plt.plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], marker="o")
        # plt.plot(projection_point_on_right.x, projection_point_on_right.y, color='b', marker="*")
        #
        started = False
        for right_position in [Point(p[0], p[1]) for p in list(self.right_edge.coords)]:

            if right_position.distance(start_point) < 1.8e-5:
                # print("Start to log positions")
                # plt.plot(right_position.x, right_position.y, color='blue', marker="o")
                started = True

            if not started:
                # print("Skip point")
                # plt.plot(right_position.x, right_position.y, color='red', marker="*")
                continue
            else:
                # print("Consider point")
                # plt.plot(right_position.x, right_position.y, color='green', marker="o")
                pass

            # Project right_position to left_edge
            projected_point = self.left_edge.interpolate(self.left_edge.project(right_position))
            # Translate the right_position 2m toward the center
            line = LineString([(right_position.x, right_position.y), (projected_point.x, projected_point.y)])
            self.driving_path.append(line.interpolate(2.0))

    def plot_all(self, car_state):

        current_position = Point(car_state['pos'][0], car_state['pos'][1])

        plt.figure(1, figsize=(5, 5))
        plt.clf()

        ax = plt.gca()
        x, y = self.left_edge.coords.xy
        ax.plot(x, y, 'r-')
        x, y = self.right_edge.coords.xy
        ax.plot(x, y, 'b-')
        driving_lane = LineString([p for p in self.driving_path])
        x, y = driving_lane.coords.xy
        ax.plot(x, y, 'g-')

        # node = {
        #     'x': target_position.x,
        #     'y': target_position.y,
        #     'z': 0.3,
        #     't': target_time,
        # }

        xs = [node['x'] for node in self.script]
        ys = [node['y'] for node in self.script]
        # print("{:.2f}".format(3.1415926));
        vs = ['{:.2f}'.format(node['avg_speed'] * 3.6) for node in self.script]

        # plt.plot(xs, ys, marker='.')
        ax = plt.gca()
        for i, txt in enumerate(vs):
            ax.annotate(txt, (xs[i], ys[i]))

        plt.plot(current_position.x, current_position.y, marker="o", color="green")

        # Center around current_positions
        ax.set_xlim([current_position.x - 50, current_position.x + 50])
        ax.set_ylim([current_position.y - 50, current_position.y + 50])

        plt.draw()
        plt.pause(0.01)

    def run(self, debug=False):
        try:
            self.vehicle = Vehicle(car_model['id'])
            electrics = Electrics()
            self.vehicle.attach_sensor('electrics', electrics)

            # Connect to running beamng
            self.bng = BeamNGpy('localhost', 64256) #, home='C://Users//Alessio//BeamNG.research_unlimited//trunk')
            self.bng = self.bng.open(launch=False)

            # Put simulator in pause awaiting while planning the driving
            self.bng.pause()
            # Connect to the existing vehicle (identified by the ID set in the vehicle instance)
            self.bng.set_deterministic()  # Set simulator to be deterministic
            self.bng.connect_vehicle(self.vehicle)
            assert self.vehicle.skt

            # Get Initial state of the car. This assumes that the script is invoked after the scenario is started
            self.bng.poll_sensors(self.vehicle)
            # Compute the "optimal" driving path and program the ai_script
            self._compute_driving_path(self.vehicle.state, self.road_model['street'])

            self.script = self.road_profiler.compute_ai_script(LineString(self.driving_path), self.car_model)

            # Enforce initial car direction nad up
            start_dir = (self.vehicle.state['dir'][0], self.vehicle.state['dir'][1], self.vehicle.state['dir'][2])
            up_dir = (0, 0, 1)

            # Configure the ego car
            self.vehicle.ai_set_mode('disabled')
            # Note that set script teleports the car by default
            self.vehicle.ai_set_script(self.script, start_dir=start_dir, up_dir=up_dir)
            # Resume the simulation
            self.bng.resume()
            # At this point the controller can stop ? or wait till it is killed

            while True:
                if debug:
                    self.bng.pause()
                    self.bng.poll_sensors(self.vehicle)
                    self.plot_all(self.vehicle.state)
                    self.bng.resume()
                # Progress the simulation for some time...
                # self.bng.step(50)
                sleep(2)

        except Exception:
            # When we brutally kill this process there's no need to log an exception
            l.error("Fatal Error", exc_info=True)
        finally:
            self.bng.close()


if __name__ == '__test__':
# if __name__ == '__main__':
    from numpy import array, load
    from shapely.geometry import asLineString

    driving_path = asLineString(load('driving.npy'))
    mu = 0.8
    speed_limit_meter_per_sec = 90.0 / 3.6
    road_profiler = RoadProfiler(mu, speed_limit_meter_per_sec)
    car_model = dict()
    # Is this m/s**2
    car_model['max_acc'] = 3.0
    car_model['max_dec'] = -6.3
    ai_script = road_profiler.compute_ai_script(driving_path, car_model)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-speed', type=int, default=90, help='Speed Limit in KM/H')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    args = parser.parse_args()

    setup_logging()

    # TODO Optionally provide car and road models
    l.debug("Setting max speed to", args.max_speed)

    car_model = dict()
    # Empirically estimated from ETK800/BeamNG
    car_model['max_acc'] = 3.5   # Not m/s**2
    car_model['max_dec'] = -0.5  # Not m/s**2
    car_model['id'] = 'egovehicle'

    road_model = dict()
    # Friction coefficient
    road_model['mu'] = 0.2  # Slippery road 0.3 # 0.5 # 0.8 Normal road
    road_model['speed_limit'] = args.max_speed
    road_model['street'] = 'street_1'

    control_model = dict()
    # 10 tends to cut sharp curve, 5 might be too conservative
    control_model['discretization_factor'] = 8

    driver = Driver(car_model, road_model, control_model)
    driver.run(debug=args.debug)
