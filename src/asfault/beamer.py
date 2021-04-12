import datetime
import io
import logging as l
import os
import signal
import socket
import subprocess
import sys
from time import sleep
import time
import os
import json

from collections import defaultdict
import shapely.geometry

from jinja2 import FileSystemLoader, Environment
from shapely.geometry import box

from asfault.network import *
from asfault.plotter import CarTracer
from asfault.tests import *
from shapely.geometry import box

# Required to force BeamNGpy to reopen the socket once the process is done
from multiprocessing import Process

from self_driving.oob_monitor import OutOfBoundsMonitor
from self_driving.road_polygon import RoadPolygon
from self_driving.nvidia_prediction import NvidiaPrediction

import traceback
from typing import Tuple

from self_driving.beamng_brewer import BeamNGBrewer, BeamNGCarCameras
# maps is a global variable in the module, which is initialized to Maps()
from self_driving.beamng_tig_maps import maps, LevelsFolder
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.simulation_data import SimulationDataRecord, SimulationData
from self_driving.simulation_data_collector import SimulationDataCollector
import self_driving.utils as us
from self_driving.vehicle_state_reader import VehicleStateReader




SCENARIOS_DIR = 'scenarios'

PREFAB_FILE = 'asfault.prefab'
VEHICLE_FILE = 'vehicle.prefab'
LUA_FILE = 'asfault.lua'
DESCRIPTION_FILE = 'asfault.json'

# Make sure that we find the right folder no matter what?
TEMPLATE_PATH = os.path.join(os.path.join(__file__, os.pardir), 'beamng_templates')
TEMPLATE_ENV = Environment(loader=FileSystemLoader(TEMPLATE_PATH))

MIN_NODE_DISTANCE = 0.1

RESULT_SUCCESS = 1
RESULT_FAILURE = -1

REASON_GOAL_REACHED = 'goal_reached'
REASON_OFF_TRACK = 'off_track'
REASON_TIMED_OUT = 'timeout'
REASON_SOCKET_TIMED_OUT = 'sockettimeout'
REASON_NO_TRACE = 'notrace'
REASON_VEHICLE_DAMAGED = 'vehicledamage'


def get_scenarios_dir(test_dir):
    return os.path.join(test_dir, SCENARIOS_DIR)


def get_car_origin(test):
    start_root = test.network.get_nodes_at(test.start)
    assert start_root
    start_root = start_root.pop()
    direction = get_path_direction_list(
        test.network, test.start, test.goal, test.get_path())[0]
    if direction:
        lane = start_root.r_lanes[-1]
    else:
        lane = start_root.l_lanes[-1]

    if test.start.geom_type != 'Point':
        l.error('Point is: %s', test.start.geom_type)
        raise ValueError('Not a point!')
    l_proj = lane.abs_l_edge.project(test.start)
    r_proj = lane.abs_r_edge.project(test.start)
    l_proj = lane.abs_l_edge.interpolate(l_proj)
    r_proj = lane.abs_r_edge.interpolate(r_proj)

    crossing = LineString([l_proj, r_proj])
    origin = crossing.interpolate(0.5, normalized=True)
    return {'x': origin.x, 'y': origin.y, 'z': 0.15}


def to_normal_origin(line):
    coords = line.coords
    xdiff = coords[0][0]
    ydiff = coords[0][1]
    xdir = coords[-1][0] - xdiff
    ydir = coords[-1][1] - ydiff
    line = LineString([(0, 0), (xdir, ydir)])
    length = line.length
    xdir = xdir / length
    ydir = ydir / length
    line = LineString([(0, 0), (xdir, ydir)])
    return line


def get_car_direction(test):
    direction = get_path_direction_list(
        test.network, test.start, test.goal, test.get_path())[0]
    if direction:
        direction = -0.5
    else:
        direction = 0.5

    start_node = test.network.get_nodes_at(test.start)
    start_node = start_node.pop()
    start_spine = start_node.get_spine()
    if test.start.geom_type != 'Point':
        l.error('Point is: %s', test.start.geom_type)
        raise ValueError('Not a point!')
    proj = start_spine.project(test.start)
    dir = start_spine.interpolate(proj + direction)
    head_vec = LineString([test.start, dir])
    head_vec = to_normal_origin(head_vec)
    coord = head_vec.coords[-1]
    return {'x': coord[0], 'y': coord[1]}


def get_node_segment_coords(node, coord, idx):
    line = node.get_line(idx)
    coords = {'x': coord[0], 'y': coord[1], 'z': 0.01, 'width': line.length}
    return coords


def get_node_coords(node, last_coords=None, sealed=True):
    ret = []
    spine = node.get_spine()
    for idx, coord in enumerate(spine.coords):
        line = node.get_line(idx)
        coords_dict = {'x': coord[0], 'y': coord[1],
                       'z': 0.01, 'width': line.length}
        if last_coords:
            point_last = Point(last_coords['x'], last_coords['y'])
            point_current = Point(coords_dict['x'], coords_dict['y'])
            distance = point_last.distance(point_current)
            if distance > MIN_NODE_DISTANCE:
                ret.append(coords_dict)
        else:
            ret.append(coords_dict)
        last_coords = coords_dict
    if not sealed:
        pass
        # ret = ret[1:-1]
    return ret


def polyline_to_decalroad(polyline, widths, z=0.01):
    nodes = []
    coords = polyline.coords
    if len(coords) != len(widths):
        raise ValueError(
            'Must give as many widths as the given polyline has coords.')

    for idx, coord in enumerate(coords):
        next_coord = {'x': coord[0], 'y': coord[1],
                      'z': z, 'width': widths[idx]}
        if nodes:
            last_coord = nodes[-1]

            last_pt = Point(last_coord['x'], last_coord['y'])
            next_pt = Point(next_coord['x'], next_coord['y'])
            distance = last_pt.distance(next_pt)
            if distance > MIN_NODE_DISTANCE:
                nodes.append(next_coord)
        else:
            nodes.append(next_coord)

    return nodes


def get_street_nodes(network, root):
    coords = []
    widths = []
    last_cursor = None
    cursor = network.get_children(root)
    assert len(cursor) <= 1
    while cursor:
        cursor = cursor.pop()
        cursor_spine = cursor.get_spine()
        cursor_coords = cursor_spine.coords
        cursor_coords = cursor_coords[:-1]
        for idx, coord in enumerate(cursor_coords):
            coords.append(coord)
            line = cursor.get_line(idx)
            widths.append(line.length)
        last_cursor = cursor
        cursor = network.get_children(cursor)

    # Add the last segment's last coord, which is skipped usually to avoid
    # overlaps from segment to segment
    cursor_spine = last_cursor.get_spine()
    cursor_coords = cursor_spine.coords
    coords.append(cursor_coords[-1])
    line = last_cursor.get_front_line()
    widths.append(line.length)

    line = LineString(coords)
    nodes = polyline_to_decalroad(line, widths)
    return nodes


def prepare_streets(network):
    roots = {*network.get_nodes(TYPE_ROOT)}
    streets = []
    while roots:
        root = roots.pop()
        street = {'street_id': root.seg_id, 'nodes': [], 'position': {}}
        nodes = get_street_nodes(network, root)

        street['position'] = nodes[0]
        street['nodes'] = nodes
        streets.append(street)

    return streets


def get_divider_from_polyline(root, divider_id, line):
    divider = {'divider_id': '{}_{}'.format(root.seg_id, divider_id),
               'nodes': [], 'position': {}}

    widths = [0.1, ] * len(line.coords)
    nodes = polyline_to_decalroad(line, widths)

    divider['position'] = nodes[0]
    divider['nodes'] = nodes

    return divider


def get_intersection_dividers(cursor_spine, intersection):
    if intersection.l_lanes:
        l_edge = intersection.l_lanes[-1].abs_l_edge
    else:
        l_edge = intersection.r_lanes[0].abs_l_edge

    if intersection.r_lanes:
        r_edge = intersection.r_lanes[-1].abs_r_edge
    else:
        r_edge = intersection.l_lanes[0].abs_r_edge

    l_inter = cursor_spine.intersection(l_edge)
    r_inter = cursor_spine.intersection(r_edge)

    # Split spine at both l_ and r_inter and see which one is shorter to find out
    # whether we need to cut off the spine at the left or right edge of the
    # intersecting road segment
    l_split_beg, _ = split(cursor_spine, l_inter)
    r_split_beg, _ = split(cursor_spine, r_inter)

    if l_split_beg.length < r_split_beg.length:
        # l_inter is the clipping point for the shared area of the intersection
        # Split spine at r_inter to get rest of divider
        _, r_split_end = split(cursor_spine, r_inter)
        return l_split_beg, r_split_end
    else:
        # r_inter is the clipping point for the shared area of the intersection
        # Split spine at l_inter to get rest of divider
        _, l_split_end = split(cursor_spine, l_inter)
        return r_split_beg, l_split_end


def get_street_dividers(network, root):
    dividers = []

    coords = []
    last_cursor = None
    cursor = network.get_children(root)
    while cursor:
        cursor = cursor.pop()
        cursor_spine = cursor.get_spine()

        intersecting = network.get_segment_intersecting_nodes(cursor)
        if intersecting:
            intersection = intersecting.pop()
            before_coords, after_coords = get_intersection_dividers(
                cursor_spine, intersection)
            coords.extend(before_coords.coords)

            line = LineString(coords)
            divider = get_divider_from_polyline(root, len(dividers) + 1, line)
            dividers.append(divider)
            coords = [*after_coords.coords]
        else:
            cursor_coords = cursor_spine.coords
            cursor_coords = cursor_coords[:-1]
            coords.extend(cursor_coords)

        last_cursor = cursor
        cursor = network.get_children(cursor)

    # Add the last segment's last coord, which is skipped usually to avoid
    # overlaps from segment to segment
    cursor_spine = last_cursor.get_spine()
    cursor_coords = cursor_spine.coords
    coords.append(cursor_coords[-1])
    line = LineString(coords)
    divider = get_divider_from_polyline(root, len(dividers) + 1, line)
    dividers.append(divider)

    return dividers


def get_street_boundary(network, root, right=False):
    dividers = []

    coords = []
    last_cursor = None
    cursor = network.get_children(root)
    fmt = 'l{}'
    if right:
        fmt = 'r{}'
    while cursor:
        cursor = cursor.pop()
        if right:
            cursor_spine = cursor.get_right_edge()
            cursor_spine = cursor_spine.parallel_offset(
                c.ev.lane_width * 0.075, 'left', join_style=shapely.geometry.JOIN_STYLE.round)
        else:
            cursor_spine = cursor.get_left_edge()
            cursor_spine = cursor_spine.parallel_offset(
                c.ev.lane_width * 0.075, 'right', join_style=shapely.geometry.JOIN_STYLE.round)

        intersecting = network.get_segment_intersecting_nodes(cursor)
        if intersecting:
            intersection = intersecting.pop()
            before_coords, after_coords = get_intersection_dividers(
                cursor_spine, intersection)
            coords.extend(before_coords.coords)

            line = LineString(coords)
            divider = get_divider_from_polyline(
                root, fmt.format(len(dividers) + 1), line)
            dividers.append(divider)
            coords = [*after_coords.coords]
        else:
            if right:
                cursor_coords = cursor_spine.coords
                cursor_coords = cursor_coords[:-1]
            else:
                cursor_coords = cursor_spine.coords
                cursor_coords = list(reversed(cursor_coords[1:]))
            coords.extend(cursor_coords)

        last_cursor = cursor
        cursor = network.get_children(cursor)

    # Add the last segment's last coord, which is skipped usually to avoid
    # overlaps from segment to segment
    if right:
        cursor_spine = last_cursor.get_right_edge()
        cursor_spine = cursor_spine.parallel_offset(
            c.ev.lane_width * 0.075, 'left', join_style=shapely.geometry.JOIN_STYLE.round)
    else:
        cursor_spine = last_cursor.get_left_edge()
        cursor_spine = cursor_spine.parallel_offset(
            c.ev.lane_width * 0.075, 'right', join_style=shapely.geometry.JOIN_STYLE.round)
    cursor_coords = cursor_spine.coords
    if right:
        coords.append(cursor_coords[-1])
    else:
        coords.append(cursor_coords[0])
    line = LineString(coords)
    divider = get_divider_from_polyline(
        root, fmt.format(len(dividers) + 1), line)
    dividers.append(divider)

    return dividers


def prepare_dividers(network):
    dividers = []
    roots = {*network.get_nodes(TYPE_ROOT)}
    while roots:
        root = roots.pop()
        street_dividers = get_street_dividers(network, root)
        dividers.extend(street_dividers)
    return dividers


def prepare_boundaries(network):
    left, right = [], []
    roots = {*network.get_nodes(TYPE_ROOT)}
    while roots:
        root = roots.pop()
        left = get_street_boundary(network, root, right=False)
        right = get_street_boundary(network, root, right=True)
    return left, right


def prepare_waypoint(node, line):
    centre = line.interpolate(0.5, normalized=True)
    l_lanes_c = len(node.l_lanes)
    r_lanes_c = len(node.r_lanes)
    scale = float(l_lanes_c + r_lanes_c) / 2
    waypoint = {'waypoint_id': node.seg_id, 'x': centre.x, 'y': centre.y,
                'z': 0.01, 'scale': c.ev.lane_width * scale}
    return waypoint


def prepare_waypoints(test):
    path = test.get_path()
    if not path:
        return []

    ret = []

    path_poly = test.get_path_polyline()
    waypoint_count = math.ceil(path_poly.length / c.ex.waypoint_step)
    for idx in range(1, int(waypoint_count - 1)):
        offset = float(idx) / waypoint_count
        path_point = path_poly.interpolate(offset, normalized=True)
        box_cursor = box(path_point.x - 0.1, path_point.y - 0.1,
                         path_point.x + 0.1, path_point.y + 0.1)
        nodes = test.network.get_intersecting_nodes(box_cursor)

        if not nodes:
            continue

        if len(nodes) == 2 and test.network.is_intersecting_pair(*nodes):
            continue

        min_distance = sys.maxsize
        min_point = None
        min_node = None
        for node in nodes:
            if node not in path:
                continue
            spine = node.get_spine()
            if path_point.geom_type != 'Point':
                l.error('Point is: %s', path_point.geom_type)
                raise ValueError('Not a point!')
            spine_proj = spine.project(path_point, normalized=True)
            spine_proj = spine.interpolate(spine_proj, normalized=True)
            distance = path_point.distance(spine_proj)
            if distance < min_distance:
                min_distance = distance
                min_point = spine_proj
                min_node = node
        if not min_point:
            continue
        assert min_point

        l_lanes_c = len(min_node.l_lanes)
        r_lanes_c = len(min_node.r_lanes)

        scale = float(l_lanes_c + r_lanes_c) / 2
        scale = c.ev.lane_width * scale
        waypoint_id = '{}_{}'.format(min_node.seg_id, len(ret))
        waypoint = {'waypoint_id': waypoint_id, 'x': min_point.x,
                    'y': min_point.y,
                    'z': 0.01, 'scale': scale}
        ret.append(waypoint)

    nodes = test.network.get_nodes_at(test.goal)
    node = nodes.pop()
    l_lanes_c = len(node.l_lanes)
    r_lanes_c = len(node.r_lanes)
    scale = float(l_lanes_c + r_lanes_c) / 2
    goal_coords = {'waypoint_id': 'goal', 'x': test.goal.x, 'y': test.goal.y,
                   'z': 0.01, 'scale': c.ev.lane_width * scale}
    ret.append(goal_coords)

    return ret


def prepare_obstacles(network):
    slots = []
    for seg in network.parentage.nodes():
        pass
    return slots


def nodes_to_coords(nodes):
    coords = list()
    for node in nodes:
        coords.append([node['x'], node['y'], -28.0, node['width']])
    return coords

class TestRunner:
    def __init__(self, test, test_dir, host, port, beamng_process=None, plot=False, ctrl=None):
        self.test = test
        self.host = host
        self.port = port
        self.plot = plot
        self.test_dir = test_dir

        self.oobs = 0
        self.is_oob = False
        self.observed_obe_states = 0

        self.race_started = False
        self.too_slow = 0

        self.seg_oob_count = defaultdict(int)
        self.oob_speeds = []

        self.current_segment = None

        self.ctrl = ctrl
        self.ctrl_process = None

        if plot:
            self.tracer = CarTracer('Trace: {}'.format(self.test.test_id),
                                    self.test.network.bounds)
            self.tracer.clear_plot()
            self.tracer.start()
            self.tracer.plot_test(self.test)
        else:
            self.tracer = None

        self.states = []
        self.start_time = None
        # defined another start time to not mess start_time
        self._start_time = time.time()
        self.end_time = None
        self.total_elapsed_time = None

        self.brewer: BeamNGBrewer = None
        self.beamng_home = c.ex.beamng_home
        self.beamng_user = "Alessio"

        self.oob_tolerance = 0.95
        self.model = None
        self.model_file = "D:\\tara\\AsFault\\models\\self-driving-car-178-2020.h5"#c.ex.model_file


    def normalise_path(self, path):
        path = path.replace('\\', '/')
        path = path[path.find('levels'):]
        return path

    def check_min_speed(self, state):
        if not self.race_started:
            return False

        if state.get_speed() < c.ex.min_speed:
            self.too_slow += 1
            if self.too_slow > c.ex.standstill_threshold:
                return True
        else:
            self.too_slow = 0
        return False

    def read_lines(self):
        self.client.settimeout(30)
        buff = io.StringIO()
        data = self.client.recv(512)
        data = str(data, 'utf-8')
        buff.write(data)
        if '\n' in data:
            line = buff.getvalue().splitlines()[0]
            yield line
            buff.flush()

    def goal_reached(self, carstate):
        pos = Point(carstate.pos_x, carstate.pos_y)
        distance = pos.distance(self.test.goal)
        # l.info("Distance to Goal " + str(distance))
        return distance < c.ex.goal_distance

    def off_track(self, nodes, vehicle_state_reader):
        oob_monitor = OutOfBoundsMonitor(RoadPolygon.from_nodes(nodes), vehicle_state_reader)
        is_oob, _, _, _ = oob_monitor.get_oob_info()
        return is_oob

    def vehicle_damaged(self, carstate):
        return carstate.damage > 10

    def get_distance_series(self):
        series = []
        for state in self.states:
            series.append(state.get_centre_distance())
        return series

    def get_average_distance(self, distances):
        total = sum(distances)
        average = total / len(distances)
        return average

    def evaluate(self, result, reason):
        options = {}
        if self.states:
            distances = self.get_distance_series()
            minimum_distance = min(distances)
            average_distance = self.get_average_distance(distances)
            maximum_distance = max(distances)

            options['minimum_distance'] = minimum_distance
            options['average_distance'] = average_distance
            options['maximum_distance'] = maximum_distance
        else:
            result = RESULT_FAILURE
            reason = REASON_NO_TRACE

        exec = TestExecution(self.test, result, reason, self.states, self.oobs,
                             self.start_time, self.end_time, **options)
        exec.seg_oob_count = self.seg_oob_count
        exec.oob_speeds = self.oob_speeds

        return exec

    def set_times(self):
        self.start_time = datetime.datetime.now()
        duration = self.test.get_path_polyline().length * c.ex.failure_timeout_spm
        self.end_time = self.start_time + datetime.timedelta(seconds=duration)
        l.info('This execution is allowed to run until: %s',
               self.end_time.isoformat())

    def get_elapsed_time(self):
        now = time.time()
        return now - self._start_time

    def get_time_left(self):

        now = datetime.datetime.now()
        return self.end_time - now

    def timed_out(self):
        # TODO Define a timeout based on the timestamp of the latest car state and duration
        # TODO Define a timeout based ont the overall test execution
        return False
        #left = self.get_time_left()
        #ret = left.seconds <= 0
        #return ret


    def run(self):
        l.info('Executing Test#{} in BeamNG.drive.'.format(self.test.test_id))
        self.set_times()


        # At this point one way or another asfault.lua will be connected as client to self.server
        timeout = self.get_time_left().seconds

        # TODO Not sure why we need to repeat this 2 times...
        counter = 2

        attempt = 0
        sim = None
        condition = True
        while condition:
            attempt += 1
            if attempt == counter:
                result = "ERROR"
                reason = 'Exhausted attempts'
                break
            if attempt > 1:
                self._close()
            if attempt > 2:
                time.sleep(5)

            sim, result, reason = self._run_simulation(self.test)

            if sim.info.success:
                if sim.exception_str:
                    result = "FAIL"
                    reason = sim.exception_str
                else:
                    result = result
                    reason = reason
                condition = False

        self.end_time = datetime.datetime.now()
        execution = self.evaluate(result, reason)
        self.test.execution = execution

        return execution

    def _is_the_car_moving(self, last_state):
        """ Check if the car moved in the past 10 seconds """

        # Has the position changed
        if self.last_observation is None:
            self.last_observation = last_state
            return True

        # If the car moved since the last observation, we store the last state and move one
        if Point(self.last_observation.pos[0],self.last_observation.pos[1]).distance(Point(last_state.pos[0], last_state.pos[1])) > self.min_delta_position:
            self.last_observation = last_state
            return True
        else:
            # How much time has passed since the last observation?
            if last_state.timer - self.last_observation.timer > 10.0:
                return False
            else:
                return True

    def _run_simulation(self, the_test) -> SimulationData:
        simulations_dir = c.rg.get_simulations_path()
        if not os.path.exists(simulations_dir):
            os.makedirs(simulations_dir)

        if not self.brewer:
            self.brewer = BeamNGBrewer(beamng_home=self.beamng_home, beamng_user=self.beamng_user)
            self.vehicle = self.brewer.setup_vehicle()
            self.camera = self.brewer.setup_scenario_camera()

        # For the execution we need the interpolated points
        street = prepare_streets(self.test.network)
        nodes = nodes_to_coords(street[0]['nodes'])
        #nodes = the_test.interpolated_points


        brewer = self.brewer
        brewer.setup_road_nodes(nodes)
        beamng = brewer.beamng
        waypoint_goal = BeamNGWaypoint('waypoint_goal', us.get_node_coords(nodes[-1]))

        # TODO Make sure that maps points to the right folder !
        if self.beamng_user is not None:
            beamng_levels = LevelsFolder(os.path.join(self.beamng_user, 'levels'))
            maps.beamng_levels = beamng_levels
            maps.beamng_map = maps.beamng_levels.get_map('tig')
            # maps.print_paths()

        maps.install_map_if_needed()
        maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())

        cameras = BeamNGCarCameras()
        vehicle_state_reader = VehicleStateReader(self.vehicle, beamng, additional_sensors=cameras.cameras_array)
        brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()

        steps = brewer.params.beamng_steps
        simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        name = os.path.join(simulations_dir, 'sim_$(id)'.replace('$(id)', simulation_id))
        sim_data_collector = SimulationDataCollector(self.vehicle, beamng, brewer.decal_road, brewer.params,
                                                     vehicle_state_reader=vehicle_state_reader,
                                                     camera=self.camera,
                                                     simulation_name=name)

        # TODO: Hacky - Not sure what's the best way to set this...
        sim_data_collector.oob_monitor.tolerance = self.oob_tolerance

        sim_data_collector.get_simulation_data().start()
        try:
            #start = timeit.default_timer()
            brewer.bring_up()
            # iterations_count = int(self.test_time_budget/250)
            # idx = 0

            brewer.bring_up()
            from keras.models import load_model
            if not self.model:
                self.model = load_model(self.model_file)
            predict = NvidiaPrediction(self.model)
            iterations_count = 100000
            idx = 0
            while True:
                idx += 1
                if idx >= iterations_count:
                    raise Exception('Timeout simulation ', sim_data_collector.name)

                sim_data_collector.collect_current_data(oob_bb=False)
                last_state: SimulationDataRecord = sim_data_collector.states[-1]
                r = last_state._asdict()

                data = {
                        "timestamp": r['timer'],
                        'pos_x': r['pos'][0],
                        'pos_y': r['pos'][1],
                        'pos_z': r['pos'][2],
                        "damage": r['damage'],
                        'steering': r['steering'],
                        'g_x': r['gforces'][0],
                        'g_y': r['gforces'][1],
                        'g_z': r['gforces'][2],
                        'vel_x': r['vel'][0],
                        'vel_y': r['vel'][1],
                        'vel_z': r['vel'][2]
                }

                state = CarState.from_dict(self.test, data)

                if state not in self.states:
                    self.states.append(state)

                # Target point reached
                # if points_distance(last_state.pos, waypoint_goal.position) < 8.0:
                #     break

                # assert self._is_the_car_moving(last_state), "Car is not moving fast enough " + str(sim_data_collector.name)

                # assert not last_state.is_oob, "Car drove out of the lane " + str(sim_data_collector.name)

                if self.tracer:
                    self.tracer.update_carstate(state)

                finished = self.goal_reached(state)
                if finished:
                    l.info('Ending test due to vehicle reaching the goal.')
                    result, reason = RESULT_SUCCESS, REASON_GOAL_REACHED
                    break

                off_track = self.off_track(nodes,vehicle_state_reader)

                if off_track:
                    if not self.is_oob:
                        l.warning('New OBE Detected')

                        self.is_oob = True
                        self.oobs += 1
                        if self.current_segment:
                            seg_key = self.current_segment.key
                            self.seg_oob_count[seg_key] += 1
                        self.oob_speeds.append(state.get_speed())

                        self.observed_obe_states += 1

                        if c.ex.dont_stop_at_obe:
                            l.debug("Don't stop @ OBE enabled, keep going")
                            pass
                        elif self.observed_obe_states <= c.ex.observation_interval:
                            l.debug('Collecting observation of car going off track. %d left', (c.ex.observation_interval-self.observed_obe_states))
                            self.observed_obe_states += 1
                        else:
                            l.info('Ending test due to vehicle going off track.')
                            result, reason = RESULT_FAILURE, REASON_OFF_TRACK
                            assert not last_state.is_oob, "Car drove out of the lane " + str(sim_data_collector.name)
                    else:
                        l.debug("- Observed OBE state")
                        if c.ex.dont_stop_at_obe:
                            l.debug("Don't stop @ OBE enabled, keep going")
                            pass
                        elif self.observed_obe_states <= c.ex.observation_interval:
                            l.debug('Collecting observation of car going off track. %d left', (c.ex.observation_interval-self.observed_obe_states))
                            self.observed_obe_states += 1
                        else:
                            l.info('Ending test due to vehicle going off track (did not come back on track).')
                            result, reason = RESULT_FAILURE, REASON_OFF_TRACK
                            break
                else:
                    self.observed_obe_states = 0

                    if self.is_oob and not c.ex.dont_stop_at_obe:
                        l.info('Ending test due to vehicle going off track (came back on track).')
                        result, reason = RESULT_FAILURE, REASON_OFF_TRACK
                        break

                    self.is_oob = False
                    self.current_segment = state.get_segment()

                damaged = self.vehicle_damaged(state)
                if damaged:
                    l.info('Ending test due to vehicle taking damage.')
                    result, reason = RESULT_FAILURE, REASON_VEHICLE_DAMAGED
                    break

                standstill = self.check_min_speed(state)
                if False and standstill:
                    l.info('Ending test due to vehicle standing still.')
                    result, reason = RESULT_FAILURE, REASON_TIMED_OUT
                    break

                img = vehicle_state_reader.sensors['cam_center']['colour'].convert('RGB')
                steering_angle, throttle = predict.predict(img, last_state)
                self.vehicle.control(throttle=throttle, steering=steering_angle, brake=0)
                beamng.step(steps)

            sim_data_collector.get_simulation_data().end(success=True)
            self.total_elapsed_time = self.get_elapsed_time()
        
        except Exception as ex:
            sim_data_collector.save()
            sim_data_collector.get_simulation_data().end(success=False, exception=ex)
            traceback.print_exception(type(ex), ex, ex.__traceback__)
        finally:
            sim_data_collector.save()
            try:
                sim_data_collector.take_car_picture_if_needed()
            except:
                pass

            self.end_iteration()

        return sim_data_collector.simulation_data, result, reason

    def end_iteration(self):
        try:
            if self.brewer:
                self.brewer.beamng.stop_scenario()
        except Exception as ex:
            traceback.print_exception(type(ex), ex, ex.__traceback__)

    def close(self):
        l.info("Closing executor")
        if self.brewer:
            try:
                self.brewer.beamng.close()
            except Exception as ex:
                traceback.print_exception(type(ex), ex, ex.__traceback__)
            self.brewer = None



BEAMNG_PROCESS = None

def gen_beamng_runner_factory(level_dir, host, port, plot=False, ctrl=None):

    # Start the shared instance of BeamNG
    # TODO How to stop this?
    global BEAMNG_PROCESS

    def factory(test):
        # Make sure that all the Test Runners will share the same instance of BeamNG. If the instance is null, each
        # process will start its runner will start its own instance...
        runner = TestRunner(test, level_dir, host, port, beamng_process=BEAMNG_PROCESS, plot=plot, ctrl=ctrl)

        return runner
    return factory


def kill_beamng():
    global BEAMNG_PROCESS
    if BEAMNG_PROCESS:
        TestRunner.kill_process(BEAMNG_PROCESS)


def run_tests(tests, test_envs, plot=True):
    distributed = {}
    per_env = int(len(tests) / len(test_envs))
    for test_env in test_envs:
        distributed[test_env.test_dir] = tests[0:per_env - 1]
        if per_env < len(tests):
            tests = tests[per_env:]
    if tests:
        distributed[test_envs[0].test_dir].extend(tests)

    for test_env in test_envs:
        env_tests = distributed[test_env.test_dir]
        for test in reversed(env_tests):
            runner = TestRunner(test, test_env, plot)
            runner.run()
