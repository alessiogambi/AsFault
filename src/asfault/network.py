import itertools
import logging as l
import math
import sys
import networkx
from networkx.algorithms.shortest_paths import has_path, shortest_path, \
    shortest_path_length
from networkx.algorithms.simple_paths import all_simple_paths
from networkx.algorithms.shortest_paths import shortest_path, all_shortest_paths
from pyqtree import Index
from shapely import affinity
from shapely.geometry import Point, LineString, Polygon
from shapely.prepared import prep

from asfault import config as c

TYPE_ROOT = 'root'
TYPE_L_TURN = 'l_turn'
TYPE_R_TURN = 'r_turn'
TYPE_STRAIGHT = 'straight'

GHOST_TYPES = (TYPE_ROOT)

SLOT_COUNT = 3
SLOT_DISTANCE = 1
MIN_SLOT_DISTANCE = 5

DEFAULT_TURTLE_HEAD = ((0.0, 0.0), (0.0, 1.0))

SEG_FACTORIES = {}


def split(line, point):
    coords = list(line.coords)
    if point.geom_type != 'Point':
        l.error('Point is: %s', point.geom_type)
        raise ValueError('Not a point!')
    point_dist = line.project(point, normalized=True)
    beg_coords = []
    while coords:
        coord = Point(*coords[0])
        if coord.geom_type != 'Point':
            l.error('Point is: %s', coord.geom_type)
            raise ValueError('Not a point!')
        coord_dist = line.project(coord, normalized=True)
        if coord_dist < point_dist:
            beg_coords.append(coord)
            coords.pop(0)
        else:
            break
    if not beg_coords:
        beg_coords.append(line.coords[0])
    beg_coords.append(line.interpolate(point_dist, normalized=True))

    end_coords = [line.interpolate(point_dist, normalized=True)]
    end_coords.extend(coords)
    if len(end_coords) < 2:
        end_coords.append(line.coords[-1])

    beg = LineString(beg_coords)
    end = LineString(end_coords)
    return beg, end


def get_outer_edge(node, direction):
    if not node:
        return None

    if direction:
        assert node.r_lanes
        return node.r_lanes[-1].abs_r_edge
    else:
        assert node.l_lanes
        return node.l_lanes[-1].abs_l_edge


def split_intersection(a_dir, a_edge, b_dir, b_edge):
    if not a_edge or not b_edge:
        return None
    assert a_edge.intersects(b_edge)

    intersection = a_edge.intersection(b_edge)
    if intersection.geom_type == 'GeometryCollection':
        if intersection.geoms:
            intersection = intersection.geoms[0]
        else:
            intersection = None

    if intersection:
        a_edge_beg, a_edge_end = split(a_edge, intersection)
        b_edge_beg, b_edge_end = split(b_edge, intersection)

        a_edge = a_edge_beg if a_dir else a_edge_end
        b_edge = b_edge_end if b_dir else b_edge_beg

        return a_edge, b_edge

    return None, None


def buffer_coords(network, coords, path, directions, idx):
    # this function is gross

    last_node, last_direction, last_edge = None, None, None
    node, direction, edge = None, None, None
    next_node, next_direction, next_edge = None, None, None

    if idx > 0:
        last_node = path[idx - 1]
        last_direction = directions[idx - 1]
    node = path[idx]
    direction = directions[idx]
    if idx < len(path) - 1:
        next_node = path[idx + 1]
        next_direction = directions[idx + 1]

    assert last_node != node
    assert node != next_node

    last_edge = get_outer_edge(last_node, last_direction)
    edge = get_outer_edge(node, direction)
    next_edge = get_outer_edge(next_node, next_direction)

    if last_edge and network.is_intersecting_pair(last_node,
                                                  node) and last_edge.intersects(
            edge):
        _, edge_split = split_intersection(last_direction, last_edge, direction,
                                           edge)
        if edge_split:
            edge = edge_split

    if next_edge and network.is_intersecting_pair(next_node,
                                                  node) and next_edge.intersects(
            edge):
        edge_split, _ = split_intersection(direction, edge, next_direction,
                                           next_edge)
        if edge_split:
            edge = edge_split

    if direction:
        coords.extend(edge.coords)
    else:
        coords.extend(reversed(list(edge.coords)))


class Turtle:

    def __init__(self, pos=(0, 0), pivot=(0, 0), angle=0):
        self.pos = list(pos)
        self.pivot = list(pivot)
        self.angle = angle
        self.head_vec = LineString([*DEFAULT_TURTLE_HEAD])

    def move(self, node):
        offset_vec = LineString([(0, 0), (node.x_off, node.y_off)])
        offset_vec = affinity.rotate(offset_vec, self.angle, origin=(0, 0))

        self.pos[0] += offset_vec.coords[-1][0]
        self.pos[1] += offset_vec.coords[-1][1]
        self.angle += node.angle
        self.angle %= 360
        self.pivot = [node.x_piv, node.y_piv]

        new_head_vec = LineString([*DEFAULT_TURTLE_HEAD])
        new_head_vec = affinity.rotate(new_head_vec, self.angle, origin=(0, 0))
        new_head_vec = affinity.translate(new_head_vec, xoff=self.pos[0],
                                          yoff=self.pos[1])
        self.head_vec = new_head_vec


def find_pivot(fan, angle, pivot_off, pivot_angle):
    coords = fan.coords
    assert len(coords) > 1
    if angle > 0:
        pivot = coords[0]
        pred = coords[1]
    else:
        pivot = coords[-1]
        pred = coords[-2]

    x_diff = pivot[0] - pred[0]
    y_diff = pivot[1] - pred[1]

    x_off = pred[0] + x_diff * pivot_off
    y_off = pred[1] + y_diff * pivot_off

    pivot_vector = LineString([pivot, (x_off, y_off)])
    pivot_vector = affinity.rotate(pivot_vector, pivot_angle, origin=pivot)

    return pivot_vector.coords[-1]


def buffer_line(start_idx, end_idx, x_off, y_off, angle, rot_orig):
    width = c.ev.lane_width
    points = []
    for i in range(start_idx, end_idx + 1):
        point = (i * width, 0)
        points.append(point)

    line = LineString(points)
    line = affinity.rotate(line, angle, origin=(0, 0))
    move = Point([x_off, y_off])
    if rot_orig:
        move = affinity.rotate(move, angle, origin=(0, 0))
    line = affinity.translate(line, xoff=move.x, yoff=move.y)

    return line


def place_slots_line(line, side):
    slots = []
    offset = line.parallel_offset(SLOT_DISTANCE, side)
    slot_gap = offset.length / (SLOT_COUNT + 1)
    for i in range(SLOT_COUNT):
        gap = (i + 1) * slot_gap
        slot = offset.interpolate(gap)
        if slots:
            last_slot = slots[-1]
            if slot.distance(last_slot) < MIN_SLOT_DISTANCE:
                continue
        slots.append(slot)
    return slots


def place_slots(seg):
    left = seg.get_left_edge()
    right = seg.get_right_edge()
    seg.left_slots = place_slots_line(left, 'left')
    seg.right_slots = place_slots_line(right, 'right')


def generate_straight_factory(key, length=0.5):
    def fac_straight(seg_id, parent, rkey=key):
        l_lanes_c = len(parent.l_lanes)
        r_lanes_c = len(parent.r_lanes)

        options = {'x': 0, 'y': length, 'angle': 0}
        child = NetworkNode(seg_id, TYPE_STRAIGHT, rkey, **options)
        child.length = length

        beg_line = buffer_line(-l_lanes_c, r_lanes_c, 0, 0, 0, True)
        end_line = buffer_line(-l_lanes_c, r_lanes_c, 0, length, 0, True)
        child.manifest_lanes([beg_line, end_line], l_lanes_c, r_lanes_c)

        # place_slots(child)

        return [child]

    return fac_straight


def generate_turn_factory(key, angle=90, pivot_off=1.05, pivot_angle=0):
    if angle < 0:
        roadtype = TYPE_L_TURN
    else:
        roadtype = TYPE_R_TURN

    def fac_turn(seg_id, parent, piv_off=pivot_off, piv_ang=pivot_angle,
                 rkey=key):
        l_lanes_c = len(parent.l_lanes)
        r_lanes_c = len(parent.r_lanes)

        child = NetworkNode(seg_id, roadtype, rkey, angle=angle)

        back_line = buffer_line(-l_lanes_c, r_lanes_c, 0, 0, 0, True)
        pivot = find_pivot(back_line, angle, piv_off, piv_ang)
        lines = [back_line]
        steps = int(math.fabs(math.ceil(angle / c.ev.max_angle)))
        todo = steps
        step_angle = angle / steps
        while todo:
            line = affinity.rotate(lines[-1], step_angle, origin=pivot)
            lines.append(line)
            todo -= 1
        child.manifest_lanes(lines, l_lanes_c, r_lanes_c)

        coords = lines[-1].coords
        child.x_off = coords[l_lanes_c][0]
        child.y_off = coords[l_lanes_c][1]
        child.x_piv = pivot[0]
        child.y_piv = pivot[1]

        child.pivot_off = piv_off
        child.pivot_angle = piv_ang

        # place_slots(child)

        return [child]

    return fac_turn


def generate_turn_factories(key_fmt):
    for pivot_off in range(2, 50, 5):
        for count in range(-8, 9):
            angle = 15 * count
            if angle != 0:
                key = key_fmt.format(angle, pivot_off)
                fac_turn = generate_turn_factory(key, angle,
                                                 pivot_off=pivot_off)
                SEG_FACTORIES[key] = fac_turn


def generate_factories():
    for count in range(1, 10, 2):
        length = 10 * count
        key = 'straight_{}'.format(length)
        fac_straight = generate_straight_factory(key, length)
        SEG_FACTORIES[key] = fac_straight

    generate_turn_factories('l_turn_{}_{:06.02f}')
    generate_turn_factories('r_turn_{}_{:06.02f}')


def seg_combination_count(window):
    options = len(SEG_FACTORIES.keys())
    options = (options ** window) * (2 ** (window - 1))
    return options


generate_factories()


class Prop:
    @staticmethod
    def to_dict(prop):
        ret = dict()
        ret['proptype'] = prop.proptype
        ret['x_off'] = prop.x_off
        ret['y_off'] = prop.y_off
        ret['angle'] = prop.angle
        return ret

    @staticmethod
    def from_dict(prop_dic):
        proptype = prop_dic['proptype']
        x_off = prop_dic['x_off']
        y_off = prop_dic['y_off']
        angle = prop_dic['angle']
        ret = Prop(proptype, x_off, y_off, angle)
        return ret

    def __init__(self, proptype, x_off, y_off, angle):
        self.proptype = proptype
        self.x_off = x_off
        self.y_off = y_off
        self.angle = angle

        self.abs_x_off = 0
        self.abs_y_off = 0
        self.abs_angle = 0

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict={}):
        return self.copy()

    def copy(self):
        return Prop(self.proptype, self.x_off, self.y_off, self.angle)

    def update_abs(self, turtle):
        x = turtle.pos[0]
        y = turtle.pos[1]
        angle = turtle.angle
        pivot = (0, 0)
        point = Point(self.x_off, self.y_off)
        point = affinity.rotate(point, angle, origin=pivot)
        point = affinity.translate(point, xoff=x, yoff=y)
        self.abs_x_off = point.x
        self.abs_y_off = point.y
        self.abs_angle = angle + self.angle % 360


class Lane:
    @staticmethod
    def to_dict(lane):
        ret = dict()
        ret['lane_id'] = lane.lane_id

        l_edge = []
        for coord in lane.l_edge.coords:
            l_edge.append([coord[0], coord[1]])
        ret['l_edge'] = l_edge

        r_edge = []
        for coord in lane.r_edge.coords:
            r_edge.append([coord[0], coord[1]])
        ret['r_edge'] = r_edge

        return ret

    @staticmethod
    def from_dict(lane_dict):
        lane_id = lane_dict['lane_id']
        l_edge = LineString([*lane_dict['l_edge']])
        r_edge = LineString([*lane_dict['r_edge']])
        return Lane(lane_id, l_edge, r_edge)

    def __init__(self, lane_id, l_edge, r_edge):
        self.lane_id = lane_id

        self.l_edge = l_edge
        self.r_edge = r_edge
        self.abs_l_edge = None
        self.abs_r_edge = None
        self.abs_polygon = None

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict={}):
        return self.copy()

    def copy(self):
        return Lane(self.lane_id, LineString(self.l_edge),
                    LineString(self.r_edge))

    def update_polygon(self):
        l_poly = list(reversed(self.abs_l_edge.coords))
        r_poly = list(self.abs_r_edge.coords)
        self.abs_polygon = Polygon(l_poly + r_poly)

    def update_abs_edges(self, turtle):
        x = turtle.pos[0]
        y = turtle.pos[1]
        angle = turtle.angle
        pivot = (0, 0)

        self.abs_l_edge = affinity.rotate(self.l_edge, angle, origin=pivot)
        self.abs_l_edge = affinity.translate(self.abs_l_edge, xoff=x, yoff=y)

        self.abs_r_edge = affinity.rotate(self.r_edge, angle, origin=pivot)
        self.abs_r_edge = affinity.translate(self.abs_r_edge, xoff=x, yoff=y)

        self.update_polygon()
        self.dirty = False

    def get_edge_difference(self, own_edge, oth_edge):
        diff = 0.0
        for own_coord, oth_coord in zip(own_edge.coords, oth_edge.coords):
            coord_diff = Point(*own_coord).distance(Point(*oth_coord))
            diff += coord_diff
        return diff

    def get_difference(self, other):
        diff = self.get_edge_difference(self.l_edge, other.l_edge)
        diff += self.get_edge_difference(self.r_edge, other.r_edge)
        return diff


class NetworkNode:

    @staticmethod
    def to_dict(node):
        ret = dict()
        ret['seg_id'] = str(node.seg_id)
        ret['roadtype'] = node.roadtype
        ret['key'] = node.key
        ret['x'] = node.x_off
        ret['y'] = node.y_off
        ret['x_piv'] = node.x_piv
        ret['y_piv'] = node.y_piv
        ret['length'] = node.length
        ret['angle'] = node.angle
        ret['options'] = {**node.options}
        ret['pivot_off'] = node.pivot_off
        ret['pivot_angle'] = node.pivot_angle

        l_lanes = [Lane.to_dict(lane) for lane in node.l_lanes]
        r_lanes = [Lane.to_dict(lane) for lane in node.r_lanes]
        ret['l_lanes'] = l_lanes
        ret['r_lanes'] = r_lanes

        l_props = [Prop.to_dict(prop) for prop in node.l_props]
        r_props = [Prop.to_dict(prop) for prop in node.r_props]
        ret['l_props'] = l_props
        ret['r_props'] = r_props

        return ret

    @staticmethod
    def from_dict(dict):
        seg_id = int(dict['seg_id'])
        roadtype = dict['roadtype']
        key = dict['key']
        options = dict['options']

        options['length'] = dict['length']
        options['angle'] = dict['angle']
        options['x'] = dict['x']
        options['y'] = dict['y']
        options['x_piv'] = dict['x_piv']
        options['y_piv'] = dict['y_piv']
        options['pivot_off'] = dict['pivot_off']
        options['pivot_angle'] = dict['pivot_angle']

        node = NetworkNode(seg_id, roadtype, key, **options)

        l_lanes = [Lane.from_dict(lane) for lane in dict['l_lanes']]
        r_lanes = [Lane.from_dict(lane) for lane in dict['r_lanes']]
        l_props = [Prop.from_dict(prop) for prop in dict['l_props']]
        r_props = [Prop.from_dict(prop) for prop in dict['r_props']]

        node.l_lanes = l_lanes
        node.r_lanes = r_lanes
        node.l_props = l_props
        node.r_props = r_props

        return node

    def __init__(self, seg_id, roadtype, key, **options):
        self.seg_id = seg_id
        self.roadtype = roadtype
        self.key = key

        self.x_off = options.get('x', 0)
        self.y_off = options.get('y', 0)
        self.x_piv = options.get('x_piv', 0)
        self.y_piv = options.get('y_piv', 0)
        self.length = options.get('length', 0)
        self.angle = options.get('angle', 0) or 0
        self.pivot_off = options.get('pivot_off', 0)
        self.pivot_angle = options.get('pivot_angle', 0)

        self.options = options

        self.l_lanes = []
        self.r_lanes = []

        self.l_props = []
        self.r_props = []

        self.root = False
        self.dead = False

        self.rel_polygon = None
        self.abs_polygon = None

    def get_spine(self):
        if len(self.r_lanes) > 0:
            spine = self.r_lanes[0]
            spine = spine.abs_l_edge
        if len(self.l_lanes) > 0:
            spine = self.l_lanes[0]
            spine = spine.abs_r_edge
        return spine

    def get_line(self, index):
        lanes = list(reversed(self.l_lanes)) + self.r_lanes
        points = []
        for lane in lanes:
            points.append(lane.abs_l_edge.coords[index])
        points.append(lanes[-1].abs_r_edge.coords[index])
        return LineString(points)

    def get_front_line(self):
        return self.get_line(-1)

    def get_back_line(self):
        return self.get_line(0)

    def get_line_count(self):
        spine = self.get_spine()
        coords = spine.coords
        return len(coords)

    def manifest_lanes(self, lines, l_lane_c, r_lane_c):
        assert lines
        edges = [[] for point in lines[0].coords]
        for line in lines:
            for idx, coord in enumerate(line.coords):
                edges[idx].append(coord)

        lane_id = 1
        self.l_lanes = []
        for i in range(0, l_lane_c):
            l_edge = LineString(edges.pop(0))
            r_edge = LineString(edges[0])
            lane = Lane(lane_id, l_edge, r_edge)
            lane_id += 1
            self.l_lanes.append(lane)
        self.l_lanes = list(reversed(self.l_lanes))

        self.r_lanes = []
        for i in range(0, r_lane_c):
            l_edge = LineString(edges.pop(0))
            r_edge = LineString(edges[0])
            lane = Lane(lane_id, l_edge, r_edge)
            lane_id += 1
            self.r_lanes.append(lane)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict={}):
        return self.copy()

    def copy(self, seg_id=None):
        l_lanes = [l.copy() for l in self.l_lanes]
        r_lanes = [l.copy() for l in self.r_lanes]
        l_props = [p.copy() for p in self.l_props]
        r_props = [p.copy() for p in self.r_props]

        if not seg_id:
            seg_id = self.seg_id

        self_copy = NetworkNode(seg_id, self.roadtype,
                                self.key, **self.options)

        self_copy.x_off = self.x_off
        self_copy.y_off = self.y_off
        self_copy.x_piv = self.x_piv
        self_copy.y_piv = self.y_piv
        self_copy.angle = self.angle
        self_copy.pivot_off = self.pivot_off
        self_copy.pivot_angle = self.pivot_angle

        self_copy.l_lanes = l_lanes
        self_copy.r_lanes = r_lanes
        self_copy.l_props = l_props
        self_copy.r_props = r_props

        return self_copy

    def get_left_edge(self, abs=True):
        if self.l_lanes:
            lane = self.l_lanes[-1]
        else:
            lane = self.r_lanes[0]

        if abs:
            l_most = lane.abs_l_edge
        else:
            l_most = lane.l_edge

        return l_most

    def get_right_edge(self, abs=True):
        if self.r_lanes:
            lane = self.r_lanes[-1]
        else:
            lane = self.l_lanes[0]

        if abs:
            r_most = lane.abs_r_edge
        else:
            r_most = lane.r_edge

        return r_most

    def update_polygon(self):
        if not self.l_lanes and not self.r_lanes:
            return

        l_most = self.get_left_edge()
        r_most = self.get_right_edge()

        l_most = list(reversed(l_most.coords))
        r_most = list(r_most.coords)
        return Polygon(l_most + r_most)

    def update_abs_polygon(self):
        polygon = self.update_polygon()
        self.abs_polygon = polygon
        return self.abs_polygon

    def update_abs_slots(self, slots, turtle):
        x = turtle.pos[0]
        y = turtle.pos[1]
        angle = turtle.angle
        pivot = (0, 0)

        abs_slots = []
        for slot in slots:
            abs_slot = affinity.rotate(slot, angle, origin=pivot)
            abs_slot = affinity.translate(abs_slot, xoff=x, yoff=y)
            abs_slots.append(abs_slot)
        return abs_slots

    def update_abs(self, turtle):
        if self.roadtype not in GHOST_TYPES:
            for lane in self.l_lanes + self.r_lanes:
                lane.update_abs_edges(turtle)

            # self.abs_l_slots = self.update_abs_slots(self.l_slots, turtle)
            # self.abs_r_slots = self.update_abs_slots(self.r_slots, turtle)

            self.update_abs_polygon()

    def get_difference(self, other):
        diff = 0.0

        own_lanes = self.l_lanes + self.r_lanes
        oth_lanes = other.l_lanes + other.r_lanes

        for own_lane, oth_lane in zip(own_lanes, oth_lanes):
            diff += own_lane.get_difference(oth_lane)

        diff += math.fabs(len(self.l_lanes) - len(other.l_lanes))
        diff += math.fabs(len(self.r_lanes) - len(other.r_lanes))

        return diff

    def __hash__(self):
        return hash(self.seg_id)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.seg_id == other.seg_id

        return False

    def __str__(self):
        return '({}, {})'.format(self.seg_id, self.key)


class NetworkLayout:

    @staticmethod
    def to_dict(layout):
        ret = dict()
        bounds = list(layout.bounds.exterior.coords)
        bounds = [[point[0], point[1]] for point in bounds]
        ret['bounds'] = bounds

        nodes = {}
        for node in layout.parentage.nodes():
            nodes[str(node.seg_id)] = NetworkNode.to_dict(node)

        parentage = []
        for edge in layout.parentage.edges():
            parent = edge[0]
            child = edge[1]
            parentage.append([parent.seg_id, child.seg_id])

        reachability = []
        for edge in layout.reachability.edges():
            from_node = edge[0]
            to_node = edge[1]
            reachability.append([from_node.seg_id, to_node.seg_id])

        ret['nodes'] = nodes
        ret['parentage'] = parentage
        ret['reachability'] = reachability

        return ret

    @staticmethod
    def from_dict(dict):
        bounds = Polygon(dict['bounds'])
        layout = NetworkLayout(bounds)

        nodes_dict = dict['nodes']
        nodes = {}
        for _, node_dict in nodes_dict.items():
            nodes[int(node_dict['seg_id'])] = NetworkNode.from_dict(node_dict)

        parentage = dict['parentage']
        for edge in parentage:
            parent = nodes[edge[0]]
            child = nodes[edge[1]]
            layout.add_parentage(parent, child)

        reachability = dict['reachability']
        for edge in reachability:
            from_node = nodes[edge[0]]
            to_node = nodes[edge[1]]
            # layout.add_reachable(from_node, to_node)

        layout.update_abs()
        layout.check_reachable_intersections()

        return layout

    def __init__(self, bounds):
        self.bounds = bounds
        self.reachability = networkx.DiGraph()
        self.parentage = networkx.DiGraph()
        self.nodes = {}
        self.spindex = None

        self.inters = list()

        self.seg_id = 1
        self.absolute_version = -1

    def get_roadtype_distribution(self):
        ret = {}
        for key, segment in self.nodes.items():
            roadtype = segment.roadtype
            if roadtype in GHOST_TYPES:
                continue

            if roadtype not in ret:
                ret[roadtype] = 0
            ret[roadtype] += 1
        return ret

    def get_segment_distribution(self):
        ret = {}
        for key, segment in self.nodes.items():
            roadtype = segment.roadtype
            if roadtype in GHOST_TYPES:
                continue
            roadkey = None
            if roadtype == TYPE_STRAIGHT:
                roadkey = '{}_{:02}'.format(roadtype, int(segment.length))
            if roadtype == TYPE_L_TURN or roadtype == TYPE_R_TURN:
                roadkey = '{}_{:02}_{:02}_{:06.2f}'.format(roadtype,
                                                           int(segment.angle),
                                                           int(
                                                               segment.pivot_angle),
                                                           segment.pivot_off)

            if roadkey not in ret:
                ret[roadkey] = 0
            ret[roadkey] += 1
        return ret

    def next_seg_id(self):
        ret = self.seg_id
        self.seg_id += 1
        return ret

    def add_node(self, node):
        self.parentage.add_node(node)
        self.reachability.add_node(node)
        self.nodes[node.seg_id] = node

    def add_parentage(self, parent, child):
        assert isinstance(parent, NetworkNode)
        assert isinstance(child, NetworkNode)
        self.parentage.add_edge(parent, child)
        self.add_reachable(parent, child)
        self.nodes[parent.seg_id] = parent
        self.nodes[child.seg_id] = child

    def add_reachable(self, from_node, to_node):
        self.reachability.add_edge(from_node, to_node, direction=True)
        self.reachability.add_edge(to_node, from_node, direction=False)
        self.nodes[from_node.seg_id] = from_node
        self.nodes[to_node.seg_id] = to_node

    def remove_node(self, node):
        self.parentage.remove_node(node)
        self.reachability.remove_node(node)
        del self.nodes[node.seg_id]

    def find_roots(self):
        ret = set()
        for node in self.parentage.nodes():
            if node.roadtype == TYPE_ROOT:
                ret.add(node)
        return ret

    def materialise_from(self, root):
        turtle = Turtle()

        todo = [root]
        while todo:
            todo_node = todo.pop(0)
            todo_node.update_abs(turtle)
            todo.extend([*self.parentage.successors(todo_node)])
            turtle.move(todo_node)

    def update_abs(self, force=False):
        if not force and self.absolute_version == self.seg_id:
            return

        roots = self.find_roots()
        for root in roots:
            self.materialise_from(root)

        self.absolute_version = self.seg_id
        self.spindex = self.build_qtree()

    def prune_oob(self):
        boundary_prep = prep(self.bounds)
        boundary_nodes = self.get_boundary_intersecting_nodes()
        for boundary_node in boundary_nodes:
            if boundary_node in self.parentage.nodes():
                children = self.get_children(boundary_node)
                for child in children:
                    if boundary_prep.disjoint(child.abs_polygon):
                        self.remove_after(boundary_node)
                        break

    def find_dead_ends(self):
        ret = set()

        for node in self.parentage.nodes():
            if node.dead:
                continue

            neighbours = self.parentage[node]
            if not neighbours:
                ret.add(node)

        return ret

    def seal_dead_ends(self):
        ends = self.find_dead_ends()
        for node in ends:
            node.dead = True

    def build_qtree(self):
        # In some cases, AsFault generates roads that go over the map boundaries and then come back. All the states
        #   outside the map are considered OBE. This is wrong and we try to fix it by defining a large buffer around
        #   the map
        from shapely.geometry import box
        from numpy import min, max

        buffer_size = 500 # TODO Not sure about the implication of this number...
        minimum = min(self.bounds.exterior.xy) - buffer_size
        maximum = max(self.bounds.exterior.xy) + buffer_size
        enlarged_bounds = box(minimum, minimum, maximum, maximum)

        bounds_prep = prep(enlarged_bounds)
        bbox = []
        for val in enlarged_bounds.bounds:
            bbox.append(val * 2)
        #ret = Index(bbox=bbox, maxdepth=100000)
        ret = Index(bbox=bbox)
        for segment in self.parentage.nodes():
            if segment.abs_polygon:
                seg_poly = segment.abs_polygon
                if not bounds_prep.disjoint(seg_poly):
                    ret.insert(segment, segment.abs_polygon.bounds)
        return ret

    def get_intersecting_nodes(self, polygon):
        ret = list()
        prepared = prep(polygon)
        others = self.spindex.intersect(polygon.bounds)
        for other in others:
            if other.roadtype in GHOST_TYPES:
                continue

            poly_other = other.abs_polygon
            if prepared.intersects(poly_other):
                ret.append(other)

        return ret

    def get_segment_intersecting_nodes(self, node):
        ret = list()
        intersecting = self.get_intersecting_nodes(node.abs_polygon)
        for intersection in intersecting:
            if intersection == node:
                continue
            if self.parentage.has_edge(intersection, node):
                continue
            if self.parentage.has_edge(node, intersection):
                continue

            ret.append(intersection)

        return ret

    def get_nodes_at(self, point):
        ret = set()
        others = self.spindex.intersect(point.bounds)
        for other in others:
            other_poly = other.abs_polygon
            if point.intersects(other_poly):
                ret.add(other)

        return ret

    def get_start_goal_candidates(self):
        boundary_nodes = self.get_boundary_intersecting_nodes()
        if len(boundary_nodes) > 1:
            options = set()
            for left, right in itertools.combinations(boundary_nodes, 2):
                _left = int(NetworkNode.to_dict(left)['seg_id'])
                _right = int(NetworkNode.to_dict(right)['seg_id'])
                if self.is_reachable(left, right) and _left < _right:
                    options.add((left, right))
                    #print(_left, _right)
                if self.is_reachable(right, left) and _right < _left:
                    options.add((right, left))
                    #print(_right, _left)
            return options

        return []

    def is_self_intersecting(self):
        for segment in self.parentage.nodes():
            if segment.roadtype in GHOST_TYPES:
                continue

            assert segment.abs_polygon
            poly_seg = segment.abs_polygon
            intersecting = self.get_intersecting_nodes(poly_seg)
            for other in intersecting:
                if segment == other:
                    continue
                if self.parentage.has_edge(segment, other):
                    continue
                if self.parentage.has_edge(other, segment):
                    continue

                return True
        return False

    def get_point_side(self, line, point):
        side = (point[0] - line.coords[0][0]) * \
               (line.coords[-1][1] - line.coords[0][1]) - \
               (point[1] - line.coords[0][1]) * \
               (line.coords[-1][0] - line.coords[0][0])

        if side < 0:
            return -1
        if side > 0:
            return 1
        return 0

    def is_full_crossing(self, mom, dad):
        d_back = dad.get_back_line()
        d_front = dad.get_front_line()

        # Test if front and back lines are in the clear after crossing
        m_poly = mom.abs_polygon
        if not d_back.disjoint(m_poly):
            l.debug('Dad back not disjoint from mom poly.')
            return False
        if not d_front.disjoint(m_poly):
            l.debug('Dad front not disjoint from mom poly.')
            return False

        m_spine = mom.get_spine()
        d_spine = dad.get_spine()

        if not m_spine.intersects(d_spine):
            l.debug('Spines dont intersect')
            return False

        intersection = m_spine.intersection(d_spine)
        if intersection.geom_type != 'Point':
            l.debug('Intersection is not a point.')
            return False

        m_straight = LineString([m_spine.coords[0], m_spine.coords[-1]])
        d_straight = LineString([d_spine.coords[0], d_spine.coords[-1]])

        # Test if the roads actually cross
        if not m_straight.intersects(d_straight):
            pass
            # return False

        # Test if front line ends up on the other side of the road as the back line
        b_sides = {self.get_point_side(m_straight, point) for point in
                   d_back.coords}
        f_sides = {self.get_point_side(m_straight, point) for point in
                   d_front.coords}
        if len(b_sides) > 1:
            l.debug('Back sides differ among each other: %s', str(b_sides))
            # return False
        if len(f_sides) > 1:
            l.debug('Front sides differ among each other: %s', str(f_sides))
            # return False

        b_side = b_sides.pop()
        f_side = f_sides.pop()
        if b_side == f_side:
            l.debug('Front and back have same sides: %s %s', b_side, f_side)
            # return False

        return True

    def has_partial_overlaps(self):
        for node in self.parentage.nodes():
            if node.roadtype in GHOST_TYPES:
                continue

            intersecting = self.get_segment_intersecting_nodes(node)
            # intersecting = self.get_intersecting_nodes(node.abs_polygon)
            if len(intersecting) > 1:
                l.debug('Found %s intersecting nodes for %s', len(intersecting),
                        str(node))
                return True

            if intersecting:
                other = set(intersecting)
                other = other.pop()
                m_spine = node.get_spine()
                d_spine = other.get_spine()
                intersection = m_spine.intersection(d_spine)
                if intersection.geom_type != 'Point':
                    return True

            while intersecting:
                intersection = intersecting.pop()
                if intersection.roadtype in GHOST_TYPES:
                    continue

                if not self.is_full_crossing(node, intersection):
                    l.debug('Crossing between %s x %s is not full.', str(node),
                            str(intersection))
                    return True

        return False

    def branch_self_intersects(self, root):
        branch = self.get_branch_from(root)
        for segment in branch:
            if segment.roadtype in GHOST_TYPES:
                continue

            intersecting = self.get_segment_intersecting_nodes(segment)
            for intersection in intersecting:
                if intersection in branch:
                    return True

        return False

    def is_reachable(self, from_node, to_node):
        ret = has_path(self.reachability, from_node, to_node)
        return ret

    def shortest_path(self, from_node, to_node):
        path = shortest_path(self.reachability, from_node, to_node)
        return path

    def all_paths(self, from_node, to_node):
        paths = all_simple_paths(self.reachability, from_node, to_node,
                                 cutoff=len(self.parentage.nodes()))
        return paths

    def all_shortest_paths(self, from_node, to_node):
        paths = all_shortest_paths(self.reachability, from_node, to_node)
        return paths

    def get_nodes(self, roadtype):
        ret = set()
        for node in self.parentage.nodes():
            if node.roadtype == roadtype:
                ret.add(node)
        return ret

    def get_roots(self):
        return self.get_nodes(TYPE_ROOT)

    def get_parent(self, child):
        incoming = self.parentage.in_edges(child)
        for edge in incoming:
            return edge[0]
        return None

    def get_children(self, parent):
        ret = set()
        for child in self.parentage[parent]:
            ret.add(child)
        for child in ret:
            assert isinstance(child, NetworkNode)
        return ret

    def get_root_from(self, node):
        while node.roadtype != TYPE_ROOT:
            node = self.get_parent(node)
        return node

    def get_root_distance(self, node):
        root = self.get_root_from(node)
        return shortest_path_length(self.parentage, root, node)

    def get_branch_from(self, node):
        ret = [node]
        while True:
            children = self.get_children(ret[-1])
            if len(children) == 1:
                ret.append(children.pop())
            else:
                return ret

    def get_branch_spine(self, root):
        branch = self.get_branch_from(root)
        branch = branch[1:]
        spine = [branch[0].get_spine().coords[0]]
        prepared = prep(self.bounds)
        for seg in branch:
            seg_poly = seg.abs_polygon
            if prepared.intersects(seg_poly):
                spine.extend(seg.get_spine().coords[1:])
            else:
                break
        return LineString(spine)

    def get_turtle_state_from(self, head):
        turtle = Turtle()
        root = self.get_root_from(head)
        path = shortest_path(self.parentage, root, head)
        for node in path:
            turtle.move(node)
        return turtle

    def get_boundary_intersecting_nodes(self):
        ret = set()
        boundary = self.bounds.exterior
        for node in self.parentage.nodes():
            if node.roadtype in GHOST_TYPES:
                continue

            spine = node.get_spine()
            intersect = boundary.intersection(spine)

            # Patch to enable the use of Shapely 1.7.0 which is the default for Python 3.7
            req_version = (3, 6, 6)
            cur_version = sys.version_info

            if intersect.is_empty:
                l.debug("Empty intersection for node spine %s", node)
                continue


            if cur_version >= req_version:
                l.debug("WARNING: Newer Python VERSION detected %s", cur_version)

                if intersect.type == 'Point':
                    ret.add(node)
                elif intersect.type == 'MultiPoint':
                    ret.add(node)
                else:
                    if len(intersect.coords) > 0:
                        ret.add(node)
            else:
                # Original AsFault code which assumes 3.6
                if intersect.type == 'Point':
                    ret.add(node)
                elif len(intersect.geoms) > 0:
                    ret.add(node)

        return ret

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict={}):
        return self.copy()

    def copy(self):
        ret = NetworkLayout(self.bounds)
        ret.seg_id = self.seg_id

        nodes = {}
        for node in self.parentage.nodes():
            node_copy = node.copy()
            nodes[node.seg_id] = node_copy

        for edge in self.parentage.edges():
            parent = nodes[edge[0].seg_id]
            child = nodes[edge[1].seg_id]
            ret.add_parentage(parent, child)

        for from_node, to_node, data in self.reachability.edges(data=True):
            from_node = nodes[from_node.seg_id]
            to_node = nodes[to_node.seg_id]
            # ret.add_reachable(from_node, to_node)

        for edge in self.parentage.edges():
            assert edge in ret.parentage.edges()

        for edge in self.reachability.edges():
            pass
            # assert edge in ret.reachability.edges()

        ret.update_abs()
        ret.check_reachable_intersections()

        return ret

    def is_intersecting_pair(self, anode, bnode):
        if anode == bnode:
            return False
        if self.parentage.has_edge(anode, bnode):
            return False
        if self.parentage.has_edge(bnode, anode):
            return False

        anode_poly = anode.abs_polygon
        intersecting = self.get_intersecting_nodes(anode_poly)
        if bnode in intersecting:
            aspine = anode.get_spine()
            bspine = bnode.get_spine()
            return aspine.intersects(bspine)

        return False

    def remove_branch_intersections(self):
        remove = []
        for from_node, to_node in self.reachability.edges():
            from_root = self.get_root_from(from_node)
            to_root = self.get_root_from(to_node)
            if from_root != to_root:
                remove.append((from_node, to_node))
        for from_node, to_node in remove:
            self.reachability.remove_edge(from_node, to_node)

    def seg_in_bounds(self, seg):
        seg_poly = seg.abs_polygon
        return not self.bounds.disjoint(seg_poly)

    def check_reachable_intersections(self):
        self.inters = list()

        for segment in self.parentage.nodes():
            if segment.roadtype in GHOST_TYPES:
                continue

            if not self.seg_in_bounds(segment):
                continue

            assert segment.abs_polygon
            poly_seg = segment.abs_polygon
            intersecting = self.get_intersecting_nodes(poly_seg)
            for other in intersecting:
                if segment == other:
                    continue
                if self.parentage.has_edge(segment, other):
                    continue
                if self.parentage.has_edge(other, segment):
                    continue

                own_spine = segment.get_spine()
                oth_spine = other.get_spine()
                intersection = own_spine.intersection(oth_spine)
                if intersection.geom_type != 'Point':
                    continue

                if segment.seg_id in self.nodes and other.seg_id in self.nodes:
                    rel_seg = self.nodes[segment.seg_id]
                    rel_oth = self.nodes[other.seg_id]
                    self.add_reachable(rel_oth, rel_seg)
                    self.add_reachable(rel_seg, rel_oth)
                    self.inters.append((rel_seg, rel_oth))

        return None

    def has_other_reachable(self, own_nodes, other_nodes):
        for own_node in own_nodes:
            for other_node in other_nodes:
                if self.reachability.has_edge(own_node, other_node):
                    return True
                if self.reachability.has_edge(other_node, own_node):
                    return True
        return False

    def merge(self, other):
        self.remove_branch_intersections()
        own_nodes = {node for node in self.nodes.values()}
        other_nodes = {}

        for node in other.parentage.nodes():
            node_copy = node.copy(seg_id=self.next_seg_id())
            other_nodes[node.seg_id] = node_copy

        for edge in other.parentage.edges():
            parent = other_nodes[edge[0].seg_id]
            child = other_nodes[edge[1].seg_id]
            self.add_parentage(parent, child)

        for edge in other.reachability.edges():
            from_node = other_nodes[edge[0].seg_id]
            to_node = other_nodes[edge[1].seg_id]
            # self.add_reachable(from_node, to_node)

        self.update_abs()
        self.prune_oob()

        self.check_reachable_intersections()

    def remove_after(self, cut_point):
        todo = {*self.get_children(cut_point)}
        while todo:
            node = todo.pop()
            todo.update(self.get_children(node))
            self.remove_node(node)

    def cut_branch(self, cut_point):
        pre = self.copy()
        pre_cut_point = pre.nodes[cut_point.seg_id]
        pre.remove_after(pre_cut_point)
        return pre, pre_cut_point

    def insert_nodes_from(self, joint, other, start, other_nodes):
        last_cursor = joint
        cursor = start
        while True:
            cursor_copy = cursor.copy(seg_id=self.next_seg_id())
            other_nodes[cursor.seg_id] = cursor_copy
            self.add_parentage(last_cursor, cursor_copy)
            children = other.get_children(cursor)
            last_cursor = cursor_copy
            if children:
                cursor = children.pop()
            else:
                break

        return other_nodes

    def join(self, joint, other, other_joint):
        self.remove_branch_intersections()
        children = self.get_children(joint)
        assert not children

        join_root = other.get_root_from(other_joint)

        other_nodes = {}
        other_nodes = self.insert_nodes_from(joint, other, other_joint,
                                             other_nodes)

        other_roots = other.get_roots()
        for root in other_roots:
            if root == join_root:
                continue
            other_root = root.copy(seg_id=self.next_seg_id())
            self.add_node(other_root)
            start = other.get_children(root).pop()
            other_nodes = self.insert_nodes_from(other_root, other, start,
                                                 other_nodes)

        # self.add_parentage(joint, other_nodes[other_joint.seg_id])
        self.update_abs()

        self.prune_oob()
        self.check_reachable_intersections()

    def replace_node(self, target, replacement):
        self.remove_branch_intersections()
        parent = self.get_parent(target)
        children = self.get_children(target)
        self.remove_node(target)
        self.add_parentage(parent, replacement)
        for child in children:
            self.add_parentage(replacement, child)
        self.update_abs(force=True)
        self.check_reachable_intersections()

    def get_difference(self, other):
        own_roots = self.get_roots()
        oth_roots = other.get_roots()

        diff = 0.0
        for own_root, oth_root in zip(own_roots, oth_roots):
            own_x_off = own_root.x_off
            own_y_off = own_root.y_off
            own_offset = Point(own_x_off, own_y_off)

            oth_x_off = oth_root.x_off
            oth_y_off = oth_root.y_off
            oth_offset = Point(oth_x_off, oth_y_off)

            diff += own_offset.distance(oth_offset)
            diff += math.fabs(own_root.angle - oth_root.angle)

            own_branch = self.get_branch_from(own_root)
            oth_branch = other.get_branch_from(oth_root)

            for own_node, oth_node in zip(own_branch, oth_branch):
                diff += own_node.get_difference(oth_node)

            diff += math.fabs(len(own_branch) - len(oth_branch))
        diff += math.fabs(len(own_roots) - len(oth_roots))

        return diff

    def has_connected_boundary_segments(self):
        candidates = self.get_start_goal_candidates()
        l.debug('Got %s start/goal candidates.', len(candidates))
        roots = self.get_nodes(TYPE_ROOT)
        boundary = self.get_boundary_intersecting_nodes()
        our_bounds = self.bounds.bounds
        for root in roots:
            street = self.get_branch_from(root)
            beg = street[1]
            if beg not in boundary:
                return False

            end = street[-1]
            end_poly = end.abs_polygon
            end_bounds = end_poly.bounds
            if end_bounds[0] >= our_bounds[0] and end_bounds[1] >= our_bounds[
                1] and end_bounds[2] <= our_bounds[2] and end_bounds[3] <= \
                    end_bounds[3]:
                l.debug('Branche\'s end lies within bounds.')
                return False

            if len(boundary) < 2:
                l.debug('Branch does not two boundary intersecting nodes!')
                return False

        ret = len(candidates) > 0
        l.debug('Has connected boundary segments: %s', ret)
        return ret

    def check_parentage(self):
        for node in self.parentage.nodes():
            children = list(self.parentage[node])
            if len(children) > 1:
                l.debug('! Node has more than one child: %s', node)
                return False
        return True

    def is_consistent(self):
        l.debug('Starting consistency check.')
        self.update_abs()

        l.debug('Checking for self-intersecting branches.')
        roots = self.get_roots()
        for root in roots:
            if self.branch_self_intersects(root):
                l.debug('Found self-intersecting branch starting at: %s',
                        str(root))
                return False
        l.debug('No self-intersecting branches found.')

        l.debug('Testing for partially overlapping segments.')
        if self.has_partial_overlaps():
            l.debug('Found a partially overlapping pair.')
            return False

        if not self.check_parentage():
            l.info('Network has nodes with too many children!')
            return False

        l.debug('No issues found. Network considered consistent.')
        return True

    def check_branch_lengths(self):
        roots = self.get_roots()
        min_length = self.bounds.bounds[2] - self.bounds.bounds[0]
        for root in roots:
            spine = self.get_branch_spine(root)
            if spine.length < min_length:
                l.debug('Spine starting at %s is too short: %s < %s.', root,
                       spine.length, min_length)
                return False

        return True

    def all_branches_connected(self):
        roots = self.get_nodes(TYPE_ROOT)
        l.debug('Got roots')
        if len(roots) > 1:
            for root in roots:
                l.debug('Checking root %s', root)
                street = self.get_branch_from(root)
                clear = False
                for seg in street:
                    if seg.roadtype in GHOST_TYPES:
                        continue

                    intersecting = self.get_segment_intersecting_nodes(seg)
                    if intersecting:
                        clear = True
                        l.debug('Found an intersection for %s', root)
                        break

                if not clear:
                    l.debug('Not all branches are connected!')
                    return False

        l.debug('All branches are connected!')
        return True

    def clean_intersection_check(self):
        try:
            for a_inter, b_inter in self.inters:
                path = [a_inter, b_inter]
                for a_dir in (False, True):
                    for b_dir in (False, True):
                        buffer_coords(self, [], path, (a_dir, b_dir), 0)
                        buffer_coords(self, [], path[::-1], (b_dir, a_dir), 0)
        except Exception as e:
            return False
        return True

    def complete_is_consistent(self):
        if not self.is_consistent():
            return False

        if not self.clean_intersection_check():
            l.debug('intersections broken!')
            return False

        if not self.has_connected_boundary_segments():
            l.debug('No two boundary segments are reachable.')
            return False

        if not self.all_branches_connected():
            l.debug('Not all branches are reachable from each other.')
            return False

        if not self.check_branch_lengths():
            l.debug('Not all branches are long enough')
            return False

        l.debug('Network is completely consistent.')
        return True
