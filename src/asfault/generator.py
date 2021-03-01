import logging as l
import random

from asfault import config as c
from asfault.network import *

OPPOSITE_PRECISION = 10000
MAX_GOAL_ANGLE = 179
ANCHOR_OFFSET = 0.05


def generate_networks(bounds, seeds):
    ret = []
    while seeds:
        seed = seeds.pop()
        gen = RoadGenerator(bounds, seed)
        while gen.grow() != RoadGenerator.done:
            continue
        ret.append(gen.network)
    return ret


def dummy_lanes(start_id, count):
    ret = []
    for lane_id in range(start_id, start_id + count):
        lane = Lane(lane_id, LineString(
            [(0.0, 0.0), (0.0, 0.0)]), LineString([(0.0, 0.0), (0.0, 0.0)]))
        ret.append(lane)
    return ret


class RoadGenerator:
    grown = 2
    done = 0
    shrank = 1

    def __init__(self, bounds):
        self.extension_stack = []
        self.available_ext = {}
        self.last_ext_key = {}

        self.network = NetworkLayout(bounds)
        self.root, self.start, self.goal = self.place_root(bounds)
        self.network.add_node(self.root)

    def place_root(self, bounds):
        start, goal = self.random_start_goal(bounds)
        centre = bounds.centroid
        x_diff = centre.x - start.x
        y_diff = centre.y - start.y
        angle = math.degrees(math.atan2(y_diff, x_diff)) - 90
        root_id = self.next_seg_id()
        options = {'x': start.x, 'y': start.y, 'angle': angle}
        root = NetworkNode(root_id, TYPE_ROOT, TYPE_ROOT, **options)
        l_lanes = dummy_lanes(1, c.ev.l_lanes)
        r_lanes = dummy_lanes(c.ev.l_lanes + 1, c.ev.r_lanes)
        root.l_lanes = l_lanes
        root.r_lanes = r_lanes
        root_point = Point(root.x_off, root.y_off)
        distance = root_point.distance(self.network.bounds)
        first_fac = generate_straight_factory(
            'straight_{}'.format(distance * 2), distance * 2)
        self.available_ext[root] = {'first': first_fac}
        return root, start, goal

    def random_start_goal(self, bounds):
        boundary_line = LineString(bounds.exterior)
        anchor_inter = random.uniform(0.0, 1.0)
        anchor = boundary_line.interpolate(anchor_inter, normalized=True)
        centre = bounds.centroid
        anchor_vec = anchor - centre
        anchor_vec = LineString([centre, anchor_vec])
        root_pos = anchor_vec.interpolate(ANCHOR_OFFSET, normalized=True)
        start = affinity.translate(anchor, xoff=root_pos.x, yoff=root_pos.y)

        anchor_inter = int((anchor_inter + 0.5) * OPPOSITE_PRECISION)
        anchor_inter = anchor_inter % OPPOSITE_PRECISION
        anchor_inter = anchor_inter / OPPOSITE_PRECISION
        goal = boundary_line.interpolate(anchor_inter, normalized=True)

        return start, goal

    def generate_factories(self):
        ext_fac = {}

        straight_fac = []
        for count in range(1, 30, 2):
            length = 15 * count
            fac_straight = generate_straight_factory(length)
            straight_fac.append(fac_straight)
        ext_fac['straight'] = straight_fac

        l_turn_fac = []
        for pivot_off in range(2, 50, 5):
            for count in range(1, 9):
                angle = -15 * count
                fac_turn = generate_turn_factory(angle, pivot_off=pivot_off)
                l_turn_fac.append(fac_turn)
        ext_fac['l_turn'] = l_turn_fac

        r_turn_fac = []
        for pivot_off in range(2, 50, 5):
            for count in range(1, 9):
                angle = 15 * count
                fac_turn = generate_turn_factory(angle, pivot_off=pivot_off)
                r_turn_fac.append(fac_turn)
        ext_fac['r_turn'] = r_turn_fac

        return ext_fac

    def next_seg_id(self):
        return self.network.next_seg_id()

    def extend(self, ext_point):
        l.debug('Attempting to extend from point: %s', str(ext_point))
        if ext_point not in self.available_ext:
            self.available_ext[ext_point] = {**SEG_FACTORIES}

        available = self.available_ext[ext_point]
        if available:
            fac_key, factory = random.sample(available.items(), 1)[0]
            l.debug('Extending with: %s', fac_key)
            self.last_ext_key[ext_point] = fac_key
            assert factory
            extensions = factory(self.next_seg_id(), ext_point)
            return extensions
        else:
            return None

    def seg_oob(self, segment):
        line = segment.get_front_line()
        return line.disjoint(self.network.bounds)

    def seal_boundaries(self):
        ext_points = self.network.find_dead_ends()
        l.debug('Found %s points that might be out of bounds.', len(ext_points))
        oob = [ext for ext in ext_points if self.seg_oob(ext)]
        oob = [ext for ext in oob if not ext.dead]
        l.debug('Ended up with %s points that are out of bounds.', len(oob))
        for seg in oob:
            seg.dead = True
            l.debug('Sealing off %s as a dead end.', str(seg))

    def is_goal_oriented(self, frontier):
        target = (self.goal.x - self.start.x, self.goal.y - self.start.y)
        for parent, segments in frontier.items():
            for segment in segments:
                if segment == self.root:
                    continue

                head_vec = self.network.get_turtle_state_from(
                    segment).head_vec.coords
                head_dir = (head_vec[-1][0] - head_vec[0]
                            [0], head_vec[-1][1] - head_vec[0][1])
                x_diff = target[0] - head_dir[0]
                y_diff = target[1] - head_dir[1]
                angle = math.degrees(math.atan2(y_diff, x_diff))
                angle = math.fabs(angle)
                if angle >= MAX_GOAL_ANGLE:
                    return False
        return True

    def get_goal_distance(self, node):
        front = node.get_front_line()
        front_centre = front.interpolate(0.5, normalized=True)
        goal_line = LineString([front_centre, self.goal])
        return goal_line.length

    def is_closer_to_goal(self, frontier):
        for parent, segments in frontier.items():
            if parent == self.root:
                continue

            for segment in segments:
                par_dist = self.get_goal_distance(parent)
                seg_dist = self.get_goal_distance(segment)
                if par_dist < seg_dist:
                    return False
        return True

    def grow(self):
        l.debug('Attempting to grow road network.')
        ext_points = self.network.find_dead_ends()
        l.debug('Found %s extension points.', len(ext_points))

        growth = {}
        for ext_point in ext_points:
            extensions = self.extend(ext_point)
            if extensions:
                growth[ext_point] = extensions
            else:
                self.shrink()
                return RoadGenerator.shrank

        if growth:
            for ext_point, extensions in growth.items():
                for extension in extensions:
                    self.network.add_parentage(ext_point, extension)
            self.extension_stack.append(growth)

            self.network.update_abs()

            if False and not self.is_goal_oriented(growth):
                l.debug('Expansion is not heading towards goal. Undoing.')
                self.shrink()
                return RoadGenerator.shrank

            if not self.is_closer_to_goal(growth):
                l.debug('Expansion did not approach goal. Undoing.')
                self.shrink()
                return RoadGenerator.shrank

            if not self.network.is_consistent():
                l.debug('Intersections found. Undoing growth.')
                self.shrink()
                return RoadGenerator.shrank
            else:
                l.debug('No intersections found, sealing off boundaries.')
                self.seal_boundaries()
                return RoadGenerator.grown

        if not self.network.check_branch_lengths():
            self.shrink()
            return RoadGenerator.shrank

        return RoadGenerator.done

    def shrink(self):
        growth = self.extension_stack.pop()
        for parent, extensions in growth.items():
            assert parent in self.last_ext_key
            assert parent in self.available_ext
            fac_key = self.last_ext_key[parent]
            available = self.available_ext[parent]
            if parent != self.root:
                todo = set()
                for key in available.keys():
                    if key[0] == fac_key[0]:
                        todo.add(key)
                for key in todo:
                    del available[key]
                #del available[fac_key]
            for extension in extensions:
                self.network.remove_node(extension)


