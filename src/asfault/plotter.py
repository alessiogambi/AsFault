import logging as l

from descartes import PolygonPatch
from matplotlib import pyplot
from matplotlib.patches import Circle
from shapely.affinity import scale
from shapely.geometry import box

from asfault.generator import dummy_lanes
from asfault.network import *
from asfault.tests import *


def clear_plot():
    pyplot.gcf().clear()


def save_plot2(out_file, dpi=90, padding=True):
    pyplot.savefig(out_file, alpha=True, dpi=dpi)


def save_plot(out_file, dpi=90, padding=True, fig_bg='white'):
    if padding:
        pyplot.savefig(out_file, alpha=False, transparent=False,
                       dpi=dpi, facecolor='#FFFFFF')
    else:
        pyplot.savefig(out_file, alpha=True, transparent=True, dpi=dpi, bbox_inches='tight',
                       pad_inches=0, facecolor='none',
                       edgecolor='none')


class TestPlotter:
    def __init__(self, plot, title, bounds_poly, plot_bounds=True):
        self.bounds_poly = bounds_poly

        self.plotted = set()
        self.plot_bounds = plot_bounds
        self.reset_plot(plot, title)

        self.widget_level = 5

    def reset_plot(self, plot, title):
        self.plot = plot
        self.title = title
        self.plot.clear()
        self.plotted.clear()
        # self.plot.set_axis_bgcolor(c.pt.colour_bg)
        self.plot.set_facecolor(c.pt.colour_bg)
        self.plot.set_title(title)

        bounds = self.bounds_poly.bounds
        xlim = (bounds[0] * c.pt.bounds_pad, bounds[2] * c.pt.bounds_pad)
        ylim = (bounds[1] * c.pt.bounds_pad, bounds[3] * c.pt.bounds_pad)

        self.plot.set_xticklabels([])
        self.plot.set_yticklabels([])
        self.plot.set_xlim(*xlim)
        self.plot.set_ylim(*ylim)

        width_edge = c.pt.factor_bounds * c.ev.lane_width
        if self.plot_bounds:
            self.plot_line(LineString(self.bounds_poly.exterior), width_edge,
                           c.pt.colour_edge)

    def plot_line(self, line, width, colour, **options):
        x_arr, y_arr = line.xy
        lines = self.plot.plot(x_arr, y_arr, color=colour, linewidth=width,
                               solid_capstyle='round', **options)
        return lines

    def plot_lane(self, lane, colour):
        l_edge = lane.abs_l_edge.coords
        r_edge = lane.abs_r_edge.coords
        if len(l_edge) > 1 and len(r_edge) > 1:
            l_edge_r = list(reversed(l_edge))
            poly = Polygon(l_edge_r + list(r_edge))
            patch = PolygonPatch(poly, fc=colour, linewidth=0)
            self.plot.add_patch(patch)
            options = {}
            if c.pt.plot_vertices:
                options['marker'] = 'o'
                options['markersize'] = c.ev.lane_width * c.pt.factor_vertex
                options['markeredgewidth'] = 0
                options['markerfacecolor'] = c.pt.colour_vertex
            width_edge = c.ev.lane_width * c.pt.factor_edge
            self.plot_line(l_edge, width_edge, c.pt.colour_edge, **options)
            self.plot_line(r_edge, width_edge, c.pt.colour_edge, **options)

    def plot_asphalt(self, polygon, colour):
        patch = PolygonPatch(polygon, fc=colour, linewidth=0)
        self.plot.add_patch(patch)

    def plot_segment(self, segment, network):
        if segment.roadtype in GHOST_TYPES:
            return

        for lane in segment.l_lanes:
            self.plot_lane(lane, c.pt.colour_lroad)
        for lane in segment.r_lanes:
            self.plot_lane(lane, c.pt.colour_rroad)

        segment.update_abs_polygon()
        if c.pt.plot_segments:
            self.add_line(LineString([*segment.abs_polygon.exterior.coords]), c.ev.lane_width * c.pt.factor_segment, None)

        if c.pt.plot_turtle:
            turtle_state = network.get_turtle_state_from(segment)
            x = turtle_state.pos[0]
            y = turtle_state.pos[1]
            self.add_circle(x, y,
                            c.ev.lane_width * c.pt.factor_turtle_head,
                            c.pt.colour_turtle)

            line = turtle_state.head_vec
            scale = c.ev.lane_width / line.length * 2
            line = affinity.scale(line, scale, scale, origin=line.coords[0])
            self.add_line(line,
                          c.ev.lane_width * c.pt.factor_turtle_line,
                          c.pt.colour_turtle)

    def get_edge_labels(self, label_fmt, edge, lane_idx):
        labels = []
        for idx, _ in enumerate(edge.coords):
            label = label_fmt.format(lane_idx + 1, idx + 1)
            labels.append(label)
        return labels

    def annotate_edge(self, label_fmt, edge, lane_idx):
        labels = self.get_edge_labels(label_fmt, edge, lane_idx)
        for label, x, y in zip(labels, edge.xy[0], edge.xy[1]):
            self.plot.annotate(
                label,
                xy=(x, y),
                xytext=(-30, 30),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.75),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

    def annotate_lane_edges(self, segment, label_fmt):
        lanes = segment.l_lanes + segment.r_lanes
        for lane_idx, lane in enumerate(lanes):
            l_edge = lane.abs_l_edge
            self.annotate_edge(label_fmt, l_edge, lane_idx)
        r_edge = lanes[-1].abs_r_edge
        self.annotate_edge(label_fmt, r_edge, len(lanes))

    def plot_topograph(self, graph, width, colour):
        for edge in graph.edges():
            from_node = edge[0]
            to_node = edge[1]

            if from_node.roadtype == TYPE_ROOT:
                continue
            if to_node.roadtype == TYPE_ROOT:
                continue

            from_node = from_node.get_spine().interpolate(0.5, normalized=True)
            to_node = to_node.get_spine().interpolate(0.5, normalized=True)

            line = LineString([from_node, to_node])
            self.add_line(line, width, colour)

    def plot_parentage(self, network):
        self.plot_topograph(network.parentage,
                            c.ev.lane_width * c.pt.factor_parent,
                            c.pt.colour_parentage)

    def plot_reachability(self, network):
        self.plot_topograph(network.reachability,
                            c.ev.lane_width * c.pt.factor_reachable,
                            c.pt.colour_reachability)

    def plot_network(self, network):
        network.update_abs()

        for segment in network.parentage.nodes():
            if segment not in self.plotted:
                self.plot_segment(segment, network)
                self.plotted.add(segment)

        if c.pt.plot_parentage:
            self.plot_parentage(network)

        if c.pt.plot_reachability:
            self.plot_reachability(network)

    def plot_test(self, test):
        self.plot_network(test.network)

        if test.start:
            self.place_start(test.start)
        if test.goal:
            self.place_goal(test.goal)

        if c.pt.plot_path_nodes and test.start and test.goal:
            path = test.get_path()
            self.trace_path(path)
        if c.pt.plot_path_line and test.start and test.goal:
            path_polyline = test.get_path_polyline()
            if c.pt.normalise_path_line:
                path_polyline = normalise_line(path_polyline)
                x_off = test.network.bounds.bounds[0]
                path_polyline = affinity.translate(
                    path_polyline, xoff=x_off, yoff=0)
            self.trace_path_polyline(test, path_polyline)
        # if test.execution:
        #     pass
            # trace = test.execution.states
            # self.plot_car_trace(trace)

    def add_circle(self, x, y, size, colour):
        node = Circle((x, y), size, fc=colour, ec='none',
                      zorder=self.widget_level)
        self.widget_level += 1
        return self.plot.add_patch(node)

    def add_line(self, line, width, colour, **options):
        x_arr, y_arr = line.xy
        self.plot.plot(x_arr, y_arr, linewidth=width, color=colour,
                       zorder=self.widget_level, **options)
        self.widget_level += 1

    def add_indicator_line(self, target, colour):
        l_centre = target.get_left_edge().interpolate(0.5, normalized=True)
        r_centre = target.get_right_edge().interpolate(0.5, normalized=True)
        line = LineString([l_centre, r_centre])
        line = scale(line, 10, 10)
        self.add_line(line, c.ev.lane_width * 0.2, colour)

    def place_start(self, point):
        self.add_circle(point.x, point.y,
                        c.ev.lane_width * c.pt.factor_start,
                        c.pt.colour_start)

    def place_goal(self, point):
        self.add_circle(point.x, point.y,
                        c.ev.lane_width * c.pt.factor_goal,
                        c.pt.colour_goal)

    def trace_path(self, path):
        recent = None
        for segment in path:
            spine = segment.get_spine()
            spine_centre = spine.interpolate(0.5, normalized=True)
            spine_next = spine.interpolate(0.55, normalized=True)
            spine = spine_centre
            if False and recent:
                line = LineString([recent, spine])
                self.add_line(line,
                              c.ev.lane_width * c.pt.factor_path_node,
                              c.pt.colour_path_nodes)
            self.add_circle(spine.x, spine.y,
                            c.ev.lane_width * c.pt.factor_path_node,
                            c.pt.colour_path_nodes)
            recent = spine

    def trace_path_polyline(self, test, path_polyline):
        self.add_line(path_polyline,
                      c.ev.lane_width * c.pt.factor_path_line,
                      c.pt.colour_path_line)

    def plot_car_trace(self, trace):
        for state in trace:
            pos = Point(state.pos_x, state.pos_y)
            self.add_circle(pos.x, pos.y,
                            c.ev.lane_width * c.pt.factor_path_node,
                            c.pt.colour_path_nodes)
            proj = state.get_path_projection()
            if False and proj:
                line = LineString([pos, proj])
                self.add_line(line,
                              c.ev.lane_width * c.pt.factor_car_state,
                              c.pt.colour_car_state)


class StandaloneTestPlotter(TestPlotter):
    def __init__(self, title, bounds_poly, figsize=(10, 10), plot_bounds=True):
        self.figure = pyplot.figure(1, figsize=figsize)
        self.figure.clear()
        self.figure.set_facecolor(c.pt.colour_bg)
        plot = self.figure.subplots(1, 1)
        self.plot_bounds = plot_bounds
        super().__init__(plot, title, bounds_poly, plot_bounds=plot_bounds)


class CarTracer(TestPlotter):
    def __init__(self, title, bounds_poly, figsize=(10, 10), **options):
        self.car_size = options.get('car_size', c.ev.lane_width)
        self.figure = pyplot.figure(1, figsize=figsize)
        self.subplot = self.figure.subplots(1, 1)

        super().__init__(self.subplot, title, bounds_poly)

    def clear_plot(self):
        self.figure.clear()
        self.figure.set_facecolor('white')
        self.subplot.clear()
        self.subplot = self.figure.subplots(1, 1)
        self.reset_plot(self.subplot, self.title)

    def start(self):
        pyplot.ion()

    def pause(self, time=0.0015):
        pyplot.pause(time)

    def update_carstate(self, state):
        pos = Point(state.pos_x, state.pos_y)
        proj = state.get_path_projection()
        self.plot.set_xlim([state.pos_x - 100, state.pos_x + 100])
        self.plot.set_ylim([state.pos_y - 100, state.pos_y + 100])
        if proj:
            line = LineString([pos, proj])
            self.add_circle(pos.x, pos.y, c.ev.lane_width *
                            0.25, c.pt.colour_car_state)
            # self.add_line(line,
            #c.ev.lane_width * c.pt.factor_car_state,
            # c.pt.colour_car_state)
            # self.add_circle(state.pos_x, state.pos_y, self.car_size, '#FF8800')


class EvolutionPlotter:
    def __init__(self, figsize=(10, 10), **options):
        self.figure = pyplot.figure(1, figsize=figsize)
        self.updaters = {}
        self.subplots = {}

        self.updaters['init_generation'] = self.init_generation
        #self.updaters['update_generation'] = self.update_generation
        self.updaters['finish_generation'] = self.finish_generation
        self.updaters['looped'] = self.looped
        self.updaters['crossedover'] = self.crossedover
        self.updaters['introduce'] = self.introduce
        self.updaters['mutated'] = self.mutated

        self.options = options

    def clear_plot(self):
        self.figure.clear()
        self.subplots.clear()

    def start(self):
        pyplot.ion()

    def pause(self, time=0.0015):
        pyplot.pause(time)

    def init_tests_display(self, tests, titles=None):
        if not titles:
            titles = ['Test#{}', ] * len(tests)

        width = int(math.ceil(math.sqrt(len(tests))))
        axarr = self.figure.subplots(width, width, squeeze=False)
        for idx, test in enumerate(tests):
            x = int(idx % width)
            y = int(idx / width)
            subplot = axarr[y, x]
            subplotter = TestPlotter(subplot, titles[idx].format(test.test_id),
                                     test.network.bounds)
            self.subplots[test.test_id] = subplotter
        self.figure.set_facecolor('white')

    def init_generation(self, evostate):
        self.clear_plot()
        self.init_tests_display(evostate[1])
        return False

    def update_plots(self, tests):
        for test in tests:
            assert test.test_id in self.subplots
            subplot = self.subplots[test.test_id]
            subplot.plot_test(test)

    def update_generation(self, evostate):
        self.clear_plot()
        self.init_tests_display(evostate[1])
        self.update_plots(evostate[1])
        return False

    def finish_generation(self, evostate):
        tests = evostate[1]
        self.clear_plot()
        self.init_tests_display(tests)
        self.update_plots(tests)

    def looped(self, evostate):
        self.clear_plot()
        tests = evostate[1]

        self.init_tests_display(tests)
        self.update_plots(tests)

    def introduce(self, evostate):
        test = evostate[1]
        self.clear_plot()
        self.init_tests_display([test], ['Introduce: Test#{}'])
        self.update_plots([test])

    def crossedover(self, evostate):
        data = evostate[1]
        mom = data[0]
        dad = data[1]
        children = data[2]
        aux = data[3]

        if children:
            tests = [mom, dad, *children]
            titles = ['Mom: Test#{}', 'Dad: Test#{}']
            for _ in tests[2:]:
                titles.append('Child: Test#{}')
        else:
            tests = [mom, dad]
            titles = ['Mom: Test#{}', 'Dad: Test#{}']
        self.clear_plot()
        self.init_tests_display(tests, titles)
        self.update_plots(tests)

        if 'aaux' in aux:
            aaux = aux['aaux']
            m_joint = aaux['m_joint']
            d_joint = aaux['d_joint']
            self.subplots[tests[0].test_id].add_indicator_line(
                m_joint, '#00FF00')
            self.subplots[tests[1].test_id].add_indicator_line(
                d_joint, '#00FF00')

        self.figure.suptitle(aux['type'].name)

    def mutated(self, evostate):
        tests = evostate[1][:2]
        aux = evostate[1][2]
        titles = ['Before: Test#{}', 'After: Test#{}']
        self.clear_plot()
        self.init_tests_display(tests, titles)
        self.update_plots(tests)

        if 'target' in aux:
            target = aux['target']
            self.subplots[tests[0].test_id].add_indicator_line(
                target, '#00FF00')

        if 'replacement' in aux:
            replacement = aux['replacement']
            self.subplots[tests[1].test_id].add_indicator_line(
                replacement, '#00FF00')

        self.figure.suptitle(aux['type'].name)

    def update(self, evostate):
        step = evostate[0]
        if step in self.updaters:
            updater = self.updaters[step]
            updater(evostate)
            return True
        return False


def get_dummy_network(size, l_lanes, r_lanes):
    bounds = box(-size, -size, size, size)
    network = NetworkLayout(bounds)
    root = {'x': 0, 'y': -size, 'angle': 0}
    root = NetworkNode(network.next_seg_id(), TYPE_ROOT, TYPE_ROOT, **root)
    l_lanes = dummy_lanes(1, l_lanes)
    r_lanes = dummy_lanes(len(l_lanes) + 1, r_lanes)
    root.l_lanes, root.r_lanes = l_lanes, r_lanes

    network.add_node(root)
    return network, bounds, root


def plot_single_lane(out_file):
    network, bounds, root = get_dummy_network(10, 0, 1)
    plotter = StandaloneTestPlotter('Single lane', bounds, plot_bounds=False)
    factory = generate_turn_factory('demo_turn', angle=60, pivot_off=5)
    turn = factory(network.next_seg_id(), root)[0]
    network.add_parentage(root, turn)
    network.update_abs(force=True)
    plotter.plot_network(network)
    save_plot(out_file, dpi=1000)


def plot_multiple_lanes(out_file):
    network, bounds, root = get_dummy_network(20, 3, 2)
    plotter = StandaloneTestPlotter('Multi lane', bounds, plot_bounds=False)
    factory = generate_turn_factory('demo_turn', angle=30, pivot_off=10)
    turn = factory(network.next_seg_id(), root)[0]
    network.add_parentage(root, turn)
    network.update_abs(force=True)
    plotter.plot_network(network)
    save_plot(out_file, dpi=1000)


def plot_vertex_order(out_file):
    network, bounds, root = get_dummy_network(15, 1, 1)
    plotter = StandaloneTestPlotter('Vertices', bounds, plot_bounds=False)
    factory = generate_turn_factory('demo_turn', angle=30, pivot_off=10)
    turn = factory(network.next_seg_id(), root)[0]
    network.add_parentage(root, turn)
    network.update_abs(force=True)
    plotter.plot_network(network)
    plotter.annotate_lane_edges(turn, '$v_{{ {},{} }}$')
    save_plot(out_file, dpi=1000)


def plot_invalid_overlap(out_file):
    network, bounds, root = get_dummy_network(20, 1, 1)

    other_root = {'x': -20, 'y': -10, 'angle': -90}
    other_root = NetworkNode(network.next_seg_id(),
                             TYPE_ROOT, TYPE_ROOT, **other_root)
    l_lanes = dummy_lanes(1, 1)
    r_lanes = dummy_lanes(2, 1)
    other_root.l_lanes, other_root.r_lanes = l_lanes, r_lanes
    network.add_node(other_root)

    plotter = StandaloneTestPlotter(
        'Invalid overlap', bounds, plot_bounds=False)
    straight = generate_straight_factory('demo_straight', 50)
    straight = straight(network.next_seg_id(), root)[0]
    network.add_parentage(root, straight)
    turn = generate_turn_factory('demo_turn', angle=90, pivot_off=3)
    turn = turn(network.next_seg_id(), other_root)[0]
    network.add_parentage(other_root, turn)
    other_turn = generate_turn_factory('other_turn', angle=90, pivot_off=5)
    other_turn = other_turn(network.next_seg_id(), turn)[0]
    network.add_parentage(turn, other_turn)
    network.update_abs(force=True)
    plotter.plot_network(network)
    save_plot(out_file, dpi=1000)


def plot_valid_overlap(out_file):
    network, bounds, root = get_dummy_network(20, 1, 1)

    other_root = {'x': -20, 'y': 0, 'angle': -90}
    other_root = NetworkNode(network.next_seg_id(),
                             TYPE_ROOT, TYPE_ROOT, **other_root)
    l_lanes = dummy_lanes(1, 1)
    r_lanes = dummy_lanes(2, 1)
    other_root.l_lanes, other_root.r_lanes = l_lanes, r_lanes
    network.add_node(other_root)

    plotter = StandaloneTestPlotter(
        'Invalid overlap', bounds, plot_bounds=False)
    straight = generate_straight_factory('demo_straight', 50)
    straight = straight(network.next_seg_id(), root)[0]
    network.add_parentage(root, straight)
    turn = generate_turn_factory('demo_turn', angle=90, pivot_off=16)
    turn = turn(network.next_seg_id(), other_root)[0]
    network.add_parentage(other_root, turn)
    network.update_abs(force=True)
    plotter.plot_network(network)
    save_plot(out_file, dpi=1000)


def plot_segment_gen(out_file):
    network, bounds, root = get_dummy_network(7, 1, 1)
    plotter = StandaloneTestPlotter('Multi lane', bounds, plot_bounds=False)
    factory = generate_turn_factory('demo_turn', angle=20, pivot_off=7)
    turn = factory(network.next_seg_id(), root)[0]
    network.add_parentage(root, turn)
    network.update_abs(force=True)
    plotter.plot_network(network)
    lines = []
    for i in range(turn.get_line_count()):
        lines.append(turn.get_line(i))
    for line in lines:
        options = {}
        if c.pt.plot_vertices:
            options['marker'] = 'o'
            options['markersize'] = c.ev.lane_width * c.pt.factor_vertex
            options['markeredgewidth'] = 0
            options['markerfacecolor'] = c.pt.colour_vertex
        #plotter.add_line(line, colour='White', width=c.ev.lane_width*0.25, **options)
    save_plot(out_file, dpi=1000)
