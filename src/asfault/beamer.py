import datetime
import io
import logging as l
import os
import signal
import socket
import subprocess
import sys
from time import sleep, time

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

from beamngpy import BeamNGpy, Scenario


SCENARIOS_DIR = 'scenarios'

PREFAB_FILE = 'asfault.prefab'
VEHICLE_FILE = 'vehicle.prefab'
LUA_FILE = 'asfault.lua'
DESCRIPTION_FILE = 'asfault.json'

TEMPLATE_PATH = 'src/asfault/beamng_templates'
TEMPLATE_ENV = Environment(loader=FileSystemLoader(TEMPLATE_PATH))

BEAMNG_BINARY = 'BeamNG.research.x64.exe'

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


def generate_test_prefab(test):
    streets = prepare_streets(test.network)
    dividers = prepare_dividers(test.network)
    l_boundaries, r_boundaries = prepare_boundaries(test.network)
    path = test.path
    if c.ex.direction_agnostic_boundary and len(path) > 1:
        beg = path[0]
        nxt = path[1]
        if test.network.parentage.has_edge(nxt, beg):
            l_boundaries, r_boundaries = r_boundaries, l_boundaries
    waypoints = prepare_waypoints(test)
    obstacles = prepare_obstacles(test.network)
    test_dict = {'start': {}, 'goal': {}}
    if 'start' in test_dict and test_dict['start']:
        test_dict['start'] = {'x': test.start.x, 'y': test.start.y, 'z': 0.01}
    else:
        test_dict['start'] = {'x': 0, 'y': 0, 'z': 0.01}

    if 'goal' in test_dict and test_dict['goal']:
        test_dict['goal'] = {'x': test.goal.x, 'y': test.goal.y, 'z': 0.01}
    else:
        test_dict['goal'] = {'x': 0, 'y': 0, 'z': 0.01}

    if c.ex.custom_beamng_template:
        l.warning("USING CUSTOM PREFAB TEMPLATE %s", c.ex.custom_beamng_template)
        prefab = TEMPLATE_ENV.get_template(c.ex.custom_beamng_template).render(streets=streets,
                                                           dividers=dividers,
                                                           l_boundaries=l_boundaries,
                                                           r_boundaries=r_boundaries,
                                                           waypoints=waypoints,
                                                           obstacles=obstacles,
                                                           test=test_dict)

    else:
        prefab = TEMPLATE_ENV.get_template(PREFAB_FILE).render(streets=streets,
                                                           dividers=dividers,
                                                           l_boundaries=l_boundaries,
                                                           r_boundaries=r_boundaries,
                                                           waypoints=waypoints,
                                                           obstacles=obstacles,
                                                           test=test_dict)
    return prefab


def generate_vehicle_prefab(test, vehicle):
    test_dict = {'start': {'x': test.start.x, 'y': test.start.y, 'z': 0.5}}
    prefab = TEMPLATE_ENV.get_template(vehicle).render(test=test_dict)
    return prefab


def generate_test_description(prefab):
    desc = TEMPLATE_ENV.get_template(DESCRIPTION_FILE).render(prefab=prefab)
    return desc


def generate_test_lua(test, **options):
    waypoints = prepare_waypoints(test)
    waypoints = ['"waypoint_{}"'.format(waypoint['waypoint_id']) for waypoint
                 in waypoints]
    waypoints = ','.join(waypoints)

    # The initial position of the car is somehow wrong as the off_track is triggered as soon as the test starts...
    pos = get_car_origin(test)
    carDir = get_car_direction(test)

    host = options.get('host', c.ex.host)
    port = options.get('port', c.ex.port)
    ai_controlled = c.ex.ai_controlled

    # BeamNG requires speed limit in m/s
    speed_limit = c.ex.speed_limit / 3.6
    speed_limit_mode = 'limit' if speed_limit > 0 else 'off'

    # This control whether BeamNG can use supernatural speed for the simulation
    max_speed = c.ex.max_speed
    navi_graph = c.ex.navi_graph
    risk = options.get('risk', c.ex.risk)
    time_left = options.get('time_left', 60)
    lua = TEMPLATE_ENV.get_template(LUA_FILE).render(test=test, host=host,
                                                     pos=pos, carDir=carDir,
                                                     max_speed=max_speed,
                                                     ai_controlled=ai_controlled,
                                                     navi_graph=navi_graph,
                                                     port=port, risk=risk,
                                                     waypoints=waypoints,
                                                     time_left=time_left,
                                                     speed_limit=speed_limit,
                                                     speed_limit_mode=speed_limit_mode)
    return lua


def write_scenario_prefab(test_dir, test):
    scenarios_dir = get_scenarios_dir(test_dir)
    prefab_file = os.path.join(scenarios_dir, ''.join(["asfault", str(test.test_id), ".prefab"]))
    prefab = generate_test_prefab(test)
    with open(prefab_file, 'w') as out:
        out.write(prefab)
    return prefab_file


def delete_scenario_prefab(test_dir, test):
    scenarios_dir = get_scenarios_dir(test_dir)
    prefab_file = os.path.join(scenarios_dir, ''.join(["asfault", str(test.test_id), ".prefab"]))
    os.remove(prefab_file)

def write_scenario_empty_prefab(test_dir):
    network = NetworkLayout(None)
    test = RoadTest(-1, network, None, None)
    return write_scenario_prefab(test_dir, test)


def write_scenario_description(test_dir, test, prefab):
    scenarios_dir = get_scenarios_dir(test_dir)
    if not os.path.exists(scenarios_dir):
        os.makedirs(scenarios_dir)
    # description_file = os.path.join(scenarios_dir, DESCRIPTION_FILE)
    description_file = os.path.join(scenarios_dir, ''.join(['asfault', str(test.test_id), '.json']))
    description = generate_test_description(prefab)
    with open(description_file, 'w') as out:
        out.write(description)
    return description_file

def delete_scenario_description(test_dir, test):
    scenarios_dir = get_scenarios_dir(test_dir)
    description_file = os.path.join(scenarios_dir, ''.join(['asfault', str(test.test_id), '.json']))
    os.remove(description_file)

def write_scenario_lua(test_dir, test, **options):
    scenarios_dir = get_scenarios_dir(test_dir)
    # lua_file = os.path.join(scenarios_dir, LUA_FILE)
    lua_file = os.path.join(scenarios_dir, ''.join(['asfault', str(test.test_id), '.lua']))
    lua = generate_test_lua(test, **options)
    with open(lua_file, 'w') as out:
        out.write(lua)
    return lua_file

def delete_scenario_lua(test_dir, test, **options):
    scenarios_dir = get_scenarios_dir(test_dir)
    lua_file = os.path.join(scenarios_dir, ''.join(['asfault', str(test.test_id), '.lua']))
    os.remove(lua_file)

def start_default_controller():
        """ Simply connect a BeamNGpy so the annoying reconnect goes away"""
        beamng = BeamNGpy('localhost', 64256)
        bng = beamng.open(launch=False)
        # Note that we cannot call stop otherwise the annoyng message will not go away...
        while True:
            sleep(10)

def load_scenario(scenario_level, scenario_name):
    """Load a scenario into a running BeamNG"""
    try:
        beamng = BeamNGpy('localhost', 64256)
        bng = beamng.open(launch=False)
        scenario = Scenario(scenario_level, scenario_name)
        scenario.bng = bng
        scenario.find(bng)
        bng.load_scenario(scenario)
        sleep(10)
        bng.start_scenario()
    finally:
        if bng:
            bng.close()

class TestRunner:
    def __init__(self, test, test_dir, host, port, beamng_process=None, plot=False, ctrl=None):
        self.test = test
        self.host = host
        self.port = port
        self.plot = plot
        self.test_dir = test_dir

        # Ensure only one server is running to accept the client?
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen()
        self.client = None

        # This identifies a running BeamNG
        self.process = beamng_process
        self.start_and_kill_beamng = self.process is None
        l.debug("Always Start and Kill BeamNG" if self.start_and_kill_beamng else "Share Running BeamNG")

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

        self.handlers = dict()
        self.handlers["HELLO"] = self.hello_handler
        self.handlers["STATE"] = self.state_handler

        # Prevent this handler to trigger ext control if we have a running instance of BeamNG. In that case we manually trigger it
        if self.start_and_kill_beamng:
            self.handlers["RACESTART"] = self.racestart_handler

        self.states = []
        self.start_time = None
        self.end_time = None

    def normalise_path(self, path):
        path = path.replace('\\', '/')
        path = path[path.find('levels'):]
        return path

    def accept_client(self, timeout=None):
        while True:
            if timeout:
                self.server.settimeout(timeout)
            con, addr = self.server.accept()
            l.debug('Accepted new client from %s', addr)
            self.client = con
            break
        return True

    @staticmethod
    def kill_process(process):
        if process:
            if os.name == 'nt':
                subprocess.call(
                    ['taskkill', '/F', '/T', '/PID', str(process.pid)])
            else:
                os.kill(process.pid, signal.SIGTERM)
            return True
        return False

    @staticmethod
    def start_shared_beamng(ctrl):

        """Start a basic instance of beamng. We will load scenarios when running the experiments"""
        # This enables the use of BeamNG.py, which means BNG will not run smoothly...
        lua = "registerCoreModule('util/researchGE')"
        userpath = c.ex.get_user_dir()
        call = [BEAMNG_BINARY, '-userpath', userpath, '-lua', lua, '-console']
        l.debug('Calling BeamNG: %s', call)
        return subprocess.Popen(call)

    def start_beamng(self, scenario_file):
        scenario_file, _ = os.path.splitext(scenario_file)
        scenario_file += '.json'
        lua = "require('scenario/scenariosLoader').startByPath('{}')"
        lua = lua.format(scenario_file)
        if self.ctrl:
            # ALESSIO: This changed since the original version of AsFault
            lua += ";registerCoreModule('util/researchGE')"
        userpath = c.ex.get_user_dir()
        call = [BEAMNG_BINARY, '-userpath', userpath, '-lua', lua, '-console']
        l.debug('Calling BeamNG: %s', call)
        self.process = subprocess.Popen(call)

    def kill_beamng(self):
        if self.process:
            TestRunner.kill_process(self.process)
        self.process = None
        return True

    def send_message(self, message):
        if not self.client:
            return

        l.debug('Sending client message: %s', message)
        message = '{}\n'.format(message)
        message = bytes(message, 'ascii')
        self.client.send(message)

    def hello_handler(self, param):
        l.debug('Got HELLO from loaded beamng scenario, responding with HELLO')
        self.send_message('HELLO:true')
        return None, None

    def racestart_handler(self, param):
        self.race_started = True
        if self.ctrl:
            self.start_controller()
            # Given enough time to asfault to move the car in the initial position while the controller plan the driving
            # TODO This is not really reliable, but I cannot find anything better...
            sleep(10)
            l.debug("Test Start")
        else:
            l.debug("Starting Dummy Controller in a Separate Process")
            self.dummy_controller = Process(target=start_default_controller, args=())
            self.dummy_controller.start()

        return None, None

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

    def state_handler(self, param):

        data = param.split(';')

        if len(data) == 11:
            data = [float(dat) for dat in data]

            state = CarState(self.test, *data)

            self.states.append(state)
            if self.tracer:
                self.tracer.update_carstate(state)

            finished = self.goal_reached(state)
            if finished:
                l.info('Ending test due to vehicle reaching the goal.')
                return RESULT_SUCCESS, REASON_GOAL_REACHED

            off_track = self.off_track(state)

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
                        return RESULT_FAILURE, REASON_OFF_TRACK
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
                        return RESULT_FAILURE, REASON_OFF_TRACK
                    pass

            else:
                self.observed_obe_states = 0

                if self.is_oob and not c.ex.dont_stop_at_obe:
                    l.info('Ending test due to vehicle going off track (came back on track).')
                    return RESULT_FAILURE, REASON_OFF_TRACK

                self.is_oob = False
                self.current_segment = state.get_segment()



            damaged = self.vehicle_damaged(state)
            if damaged:
                pass
                #l.info('Ending test due to vehicle taking damage.')
                # return RESULT_FAILURE, REASON_VEHICLE_DAMAGED

            standstill = self.check_min_speed(state)
            if False and standstill:
                l.info('Ending test due to vehicle standing still.')
                return RESULT_FAILURE, REASON_TIMED_OUT

        return None, None

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
        return distance < c.ex.goal_distance

    def off_track(self, carstate):
        distance = carstate.get_centre_distance()
        # TODO Why lane_width is under Evolution and not Execution configuration?
        if distance > c.ev.lane_width / 2.0:
            l.debug("Car is off track: distance %f", distance)

            # Car is off track
            if distance >= c.ev.lane_width / 2.0 + c.ev.tolerance:
                return True
            else:
                l.debug("Car is off track but within the tolerance value. ")
                return False

        return False

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

    def get_time_left(self):
        now = datetime.datetime.now()
        return self.end_time - now

    def timed_out(self):
        left = self.get_time_left()
        ret = left.seconds <= 0
        return ret


    def start_controller(self):
        l.debug('Calling controller process: %s', self.ctrl)
        self.ctrl_process = subprocess.Popen(self.ctrl)

    def kill_controller(self):
        l.debug('Terminating controller process.')
        if self.ctrl_process:
            TestRunner.kill_process(self.ctrl_process)
        else:
            try:
                if self.dummy_controller and self.dummy_controller.is_alive():
                    self.dummy_controller.terminate()
            except Exception as e:
                l.error("Exception while terminating dummy controller")

        self.ctrl_process = None

    def run(self):
        l.info('Executing Test#{} in BeamNG.drive.'.format(self.test.test_id))
        self.set_times()

        prefab_file = write_scenario_prefab(self.test_dir, self.test)
        prefab_file = self.normalise_path(prefab_file)

        scenario_file = write_scenario_description(self.test_dir, self.test, prefab_file)
        scenario_file = self.normalise_path(scenario_file)

        write_scenario_lua(self.test_dir, self.test, host=self.host,
                           port=self.port, time_left=self.get_time_left().seconds)

        if self.start_and_kill_beamng:
            self.start_beamng(scenario_file)
        else:

            l.debug("Loading scenario in a separate process")

            p = Process(target=load_scenario, args=('asfault',''.join(['asfault', str(self.test.test_id)])))
            p.start()
            p.join()  # this blocks until the process terminates

            # TODO I have no idea how this will play with BNG...

            # After starting the scenario we can start the external process, which in turn will use its own BeamNGpy instance...
            self.racestart_handler(None)

        # At this point one way or another asfault.lua will be connected as client to self.server
        timeout = self.get_time_left().seconds
        accepted = self.accept_client(timeout=30)

        result = None
        reason = None

        if not accepted:
            result = RESULT_FAILURE
            reason = REASON_SOCKET_TIMED_OUT


        while not result:
            for line in self.read_lines():
                split = line.find(':')
                if split != -1:
                    command = line[:split]
                    param = line[split + 1:]
                    if command in self.handlers:
                        handler = self.handlers[command]
                        result, reason = handler(param)
            if self.tracer:
                self.tracer.pause()

            if not result and self.timed_out():
                l.info('Ending test execution due to vehicle timing out.')
                result, reason = RESULT_FAILURE, REASON_TIMED_OUT
                break

        # At this point the scenario is finished

        # Kill the external controller if one is running
        self.kill_controller()

        # Kill asfault.lua
        self.send_message('KILL:0')
        sleep(0.5)


        if self.start_and_kill_beamng:
            self.kill_beamng()

        self.end_time = datetime.datetime.now()
        execution = self.evaluate(result, reason)
        self.test.execution = execution

        return execution

    def close(self):
        l.debug("Closing Server")
        ## Not sure why this is not automatically called upon finishing the execution...
        self.server.close()
        # TODO Add a flag to skip this step if we are running in debug...
        # Make sure we clean up beamng scenario folders
        self.clean_up_beamng_scenarios()

    def clean_up_beamng_scenarios(self):
        l.debug("Clean up resources")

        delete_scenario_prefab(self.test_dir, self.test)
        delete_scenario_description(self.test_dir, self.test)
        delete_scenario_lua(self.test_dir, self.test)


BEAMNG_PROCESS = None

def gen_beamng_runner_factory(level_dir, host, port, plot=False, ctrl=None):

    # Start the shared instance of BeamNG
    # TODO How to stop this?
    global BEAMNG_PROCESS

    # If we want to use BeamNG py we need to enable researchGE so BeamNG will try to connect over and over...
    retry = 0
    while True:

        if retry >= 5:
            raise Exception("Cannot start BeamNG. Giving up")

        if retry > 0:
            l.info("Wait 10 seconds before retry")
            sleep(10)
        try:
            BEAMNG_PROCESS = TestRunner.start_shared_beamng(ctrl)
            # Does this make any difference? Will BeamNG fail after some time or soon
            sleep(5)
            if BEAMNG_PROCESS.poll() is None:
                # Process is running.
                break
            else:
                l.error("BeamNG did not started. Retry")
                retry += 1

        except Exception as e:
            l.error("Cannot start BeamNG", e)
            retry += 1

    # This should ensure that BeamNG is actually started before we can instantiate the various runners...
    sleep(20)

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
