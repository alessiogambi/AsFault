import json
from pathlib import Path

import os.path

PATH_CFG = 'cfg'
PATH_OUTPUT = 'output'
PATH_PLOTS = 'plots'
PATH_GRAPHS = 'graphs'
PATH_TESTS = 'tests'
PATH_SIMULATIONS = 'simulations'
PATH_EXECS = 'execs'
PATH_REPLAYS = 'replay'
FILE_EVOLUTION = 'evolution.json'
FILE_EXECUTION = 'execution.json'
FILE_PLOT = 'plot.json'
FILE_RESULTS = 'results.csv'
FILE_OOBS_GENS = 'obes.png'

rg = None
ev = None
pt = None
ex = None


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


class AsFaultEnv:
    def __init__(self, env_dir):
        self.env_dir = env_dir

    def get_cfg_path(self):
        return os.path.join(self.env_dir, PATH_CFG)

    def get_cfg_file_path(self, name):
        cfg_path = self.get_cfg_path()
        cfg_path = os.path.join(cfg_path, name)
        return cfg_path

    def get_ev_path(self):
        return self.get_cfg_file_path(FILE_EVOLUTION)

    def get_pt_path(self):
        return self.get_cfg_file_path(FILE_PLOT)

    def get_ex_path(self):
        return self.get_cfg_file_path(FILE_EXECUTION)

    def get_output_path(self):
        return os.path.join(self.env_dir, PATH_OUTPUT)

    def get_plots_path(self):
        plots_path = self.get_output_path()
        return os.path.join(plots_path, PATH_PLOTS)

    def get_graphs_path(self):
        graphs_path = self.get_output_path()
        return os.path.join(graphs_path, PATH_GRAPHS)

    def get_tests_path(self):
        output_path = self.get_output_path()
        return os.path.join(output_path, PATH_TESTS)

    def get_simulations_path(self):
        output_path = self.get_output_path()
        return os.path.join(output_path, PATH_SIMULATIONS)

    def get_results_path(self):
        output_path = self.get_output_path()
        return os.path.join(output_path, FILE_RESULTS)

    def get_final_path(self):
        output_path = self.get_output_path()
        return os.path.join(output_path, 'final')

    def get_oobs_gens_path(self):
        graphs_path = self.get_graphs_path()
        return os.path.join(graphs_path, FILE_OOBS_GENS)

    def get_oob_segs_path(self):
        graphs_path = self.get_graphs_path()
        return os.path.join(graphs_path, 'obe_segs.png')

    def get_execs_path(self):
        execs_path = self.get_output_path()
        return os.path.join(execs_path, PATH_EXECS)

    def get_replays_path(self):
        replays_path = self.get_output_path()
        return os.path.join(replays_path, PATH_REPLAYS)

    def ensure_directories(self):
        paths = [
            self.env_dir,
            self.get_cfg_path(),
            self.get_output_path(),
            self.get_plots_path(),
            self.get_tests_path(),
            self.get_execs_path(),
            self.get_replays_path(),
            self.get_graphs_path()
        ]
        for path in paths:
            ensure_directory(path)


class EvolutionConfig:
    LANE_WIDTH = 4.0
    # Note that to claim an OBE at least half of the car is already out...
    TOLERANCE = 0.0

    L_LANES = 1
    R_LANES = 1
    MAX_ANGLE = 5.0
    BOUNDS = 1000
    POP_SIZE = 25
    MUT_CHANCE = 0.05
    INTRO_CHANCE = 0.15
    EVALUATOR = None # 'lanedist'
    SELECTOR =  None # 'tournament'
    ESTIMATOR = None # 'length'
    JOIN_PROBABILITY = 0.5 # REPLACE THIS WITH CROSSOVER_PROBABILITY
    PARTIAL_MERGE_M_COUNT = 1
    PARTIAL_MERGE_D_COUNT = 1
    TRY_ALL_OPS = True

    ATTEMPT_REPAIR = False
    SEARCH_STOPPER = None
    RESTART_SEARCH = True
    POPULATION_MERGER = None

    CROSSOVER_PROBABILITY = 0.8

    @staticmethod
    def get_default():
        ret = {}

        ret['lane_width'] = EvolutionConfig.LANE_WIDTH
        ret['tolerance'] = EvolutionConfig.TOLERANCE

        ret['l_lanes'] = EvolutionConfig.L_LANES
        ret['r_lanes'] = EvolutionConfig.R_LANES
        ret['max_angle'] = EvolutionConfig.MAX_ANGLE
        ret['bounds'] = EvolutionConfig.BOUNDS
        ret['pop_size'] = EvolutionConfig.POP_SIZE
        ret['mut_chance'] = EvolutionConfig.MUT_CHANCE
        ret['intro_chance'] = EvolutionConfig.INTRO_CHANCE
        ret['evaluator'] = EvolutionConfig.EVALUATOR
        ret['selector'] = EvolutionConfig.SELECTOR
        ret['estimator'] = EvolutionConfig.ESTIMATOR
        ret['join_probability'] = EvolutionConfig.JOIN_PROBABILITY
        ret['partial_merge_m_count'] = EvolutionConfig.PARTIAL_MERGE_M_COUNT
        ret['partial_merge_d_count'] = EvolutionConfig.PARTIAL_MERGE_D_COUNT
        ret['try_all_ops'] = EvolutionConfig.TRY_ALL_OPS

        ret['attempt_repair'] = EvolutionConfig.ATTEMPT_REPAIR
        ret['search_stopper'] = EvolutionConfig.SEARCH_STOPPER
        ret['restart_search'] = EvolutionConfig.RESTART_SEARCH

        ret['pop_merger'] = EvolutionConfig.POPULATION_MERGER
        ret['crossover_probability'] = EvolutionConfig.CROSSOVER_PROBABILITY

        return ret

    def __init__(self, path):
        with open(path, 'r') as infile:
            cfg = json.loads(infile.read())

        self.lane_width = cfg.get('lane_width', EvolutionConfig.LANE_WIDTH)
        self.tolerance = cfg.get('tolerance', EvolutionConfig.TOLERANCE)

        self.l_lanes = cfg.get('l_lanes', EvolutionConfig.L_LANES)
        self.r_lanes = cfg.get('r_lanes', EvolutionConfig.R_LANES)
        self.max_angle = cfg.get('max_angle', EvolutionConfig.MAX_ANGLE)
        self.bounds = cfg.get('bounds', EvolutionConfig.BOUNDS)
        self.pop_size = cfg.get('pop_size', EvolutionConfig.POP_SIZE)
        self.mut_chance = cfg.get('mut_chance', EvolutionConfig.MUT_CHANCE)
        self.intro_chance = cfg.get('intro_chance', EvolutionConfig.INTRO_CHANCE)
        self.evaluator = cfg.get('evaluator', EvolutionConfig.EVALUATOR)
        self.selector = cfg.get('selector', EvolutionConfig.SELECTOR)
        self.estimator = cfg.get('estimator', EvolutionConfig.ESTIMATOR)
        self.join_probability = cfg.get('join_probability', EvolutionConfig.JOIN_PROBABILITY)
        self.partial_merge_m_count = cfg.get('partial_merge_m_count', EvolutionConfig.PARTIAL_MERGE_M_COUNT)
        self.partial_merge_d_count = cfg.get('partial_merge_d_count', EvolutionConfig.PARTIAL_MERGE_D_COUNT)
        self.try_all_ops = cfg.get('try_all_ops', EvolutionConfig.TRY_ALL_OPS)

        self.attempt_repair = cfg.get('attempt_repair', EvolutionConfig.ATTEMPT_REPAIR)
        self.search_stopper = cfg.get('search_stopper', EvolutionConfig.SEARCH_STOPPER)
        self.restart_search = cfg.get('restart_search', EvolutionConfig.RESTART_SEARCH)

        self.pop_merger = cfg.get('pop_merger', EvolutionConfig.POPULATION_MERGER)
        self.crossover_probability = cfg.get('crossover_probability', EvolutionConfig.CROSSOVER_PROBABILITY)


class PlotConfig:
    DPI_INTERMEDIATE = 144
    DPI_FINAL = 300

    COLOUR_BG = '#58764c'
    COLOUR_BOUNDS = '#cccccc'
    COLOUR_LROAD = '#777777'
    COLOUR_RROAD = '#888888'
    COLOUR_EDGE = '#dddddd'
    COLOUR_PARENTAGE = '#00FF80'
    COLOUR_REACHABILITY = '#FFFF00'
    COLOUR_TURTLE = '#00FFFF'
    COLOUR_PATH_NODES = '#00FF00'
    COLOUR_PATH_LINE = '#FF88FF'
    COLOUR_START = '#0000FF'
    COLOUR_GOAL = '#FF0000'
    COLOUR_JOINT = '#FF69B4'
    COLOUR_CAR_STATE = '#FF0000'
    COLOUR_VERTEX = '#000000'
    COLOUR_SEGMENT = '#FF8800'

    FACTOR_TURTLE_HEAD = 0.25
    FACTOR_TURTLE_LINE = 0.1
    FACTOR_PARENT = 0.1
    FACTOR_REACHABLE = 0.1
    FACTOR_START = 2
    FACTOR_GOAL = 2
    FACTOR_PATH_NODE = 0.25
    FACTOR_PATH_LINE = 0.25
    FACTOR_CAR_STATE = 0.1
    FACTOR_JOINT = 4
    FACTOR_VERTEX = 0.1
    FACTOR_EDGE = 0.125
    FACTOR_BOUNDS = 0.25
    FACTOR_SEGMENT = 0.15

    BOUNDS_PAD = 1.05

    SIZE_MAP = 4096

    @staticmethod
    def get_default():
        ret = {}

        ret['dpi_intermediate'] = PlotConfig.DPI_INTERMEDIATE
        ret['dpi_final'] = PlotConfig.DPI_FINAL

        ret['colour_bg'] = PlotConfig.COLOUR_BG
        ret['colour_bounds'] = PlotConfig.COLOUR_BOUNDS
        ret['colour_lroad'] = PlotConfig.COLOUR_LROAD
        ret['colour_rroad'] = PlotConfig.COLOUR_RROAD
        ret['colour_edge'] = PlotConfig.COLOUR_EDGE
        ret['colour_parentage'] = PlotConfig.COLOUR_PARENTAGE
        ret['colour_reachability'] = PlotConfig.COLOUR_REACHABILITY
        ret['colour_turtle'] = PlotConfig.COLOUR_TURTLE
        ret['colour_path_nodes'] = PlotConfig.COLOUR_PATH_NODES
        ret['colour_path_line'] = PlotConfig.COLOUR_PATH_LINE
        ret['colour_start'] = PlotConfig.COLOUR_START
        ret['colour_goal'] = PlotConfig.COLOUR_GOAL
        ret['colour_car_state'] = PlotConfig.COLOUR_CAR_STATE
        ret['colour_vertex'] = PlotConfig.COLOUR_VERTEX
        ret['colour_segment'] = PlotConfig.COLOUR_SEGMENT

        ret['factor_turtle_head'] = PlotConfig.FACTOR_TURTLE_HEAD
        ret['factor_turtle_line'] = PlotConfig.FACTOR_TURTLE_LINE
        ret['factor_parent'] = PlotConfig.FACTOR_PARENT
        ret['factor_reachable'] = PlotConfig.FACTOR_REACHABLE
        ret['factor_start'] = PlotConfig.FACTOR_START
        ret['factor_goal'] = PlotConfig.FACTOR_GOAL
        ret['factor_path_node'] = PlotConfig.FACTOR_PATH_NODE
        ret['factor_path_line'] = PlotConfig.FACTOR_PATH_LINE
        ret['factor_car_state'] = PlotConfig.FACTOR_CAR_STATE
        ret['factor_joint'] = PlotConfig.FACTOR_JOINT
        ret['factor_vertex'] = PlotConfig.FACTOR_VERTEX
        ret['factor_edge'] = PlotConfig.FACTOR_EDGE
        ret['factor_bounds'] = PlotConfig.FACTOR_BOUNDS
        ret['factor_segment'] = PlotConfig.FACTOR_SEGMENT

        ret['bounds_pad'] = PlotConfig.BOUNDS_PAD

        ret['parentage'] = False
        ret['reachability'] = False
        ret['path_nodes'] = False
        ret['path_line'] = True
        ret['normalise_path_line'] = False
        ret['turtle'] = False
        ret['vertices'] = False
        ret['segments'] = False

        ret['size_map'] = PlotConfig.SIZE_MAP

        return ret

    def __init__(self, path):
        with open(path, 'r') as infile:
            cfg = json.loads(infile.read())

        self.dpi_intermediate = cfg.get('dpi_intermediate', PlotConfig.DPI_INTERMEDIATE)
        self.dpi_final = cfg.get('dpi_final', PlotConfig.DPI_FINAL)

        self.colour_bg = cfg.get('colour_bg', PlotConfig.COLOUR_BG)
        self.colour_bounds = cfg.get('colour_bounds', PlotConfig.COLOUR_BOUNDS)
        self.colour_lroad = cfg.get('colour_lroad', PlotConfig.COLOUR_LROAD)
        self.colour_rroad = cfg.get('colour_rroad', PlotConfig.COLOUR_RROAD)
        self.colour_edge = cfg.get('colour_edge', PlotConfig.COLOUR_EDGE)
        self.colour_parentage = cfg.get('colour_parentage',
                                        PlotConfig.COLOUR_PARENTAGE)
        self.colour_reachability = cfg.get('colour_reachability',
                                           PlotConfig.COLOUR_REACHABILITY)
        self.colour_turtle = cfg.get('colour_turtle', PlotConfig.COLOUR_TURTLE)
        self.colour_path_nodes = cfg.get('colour_path_nodes',
                                         PlotConfig.COLOUR_PATH_NODES)
        self.colour_path_line = cfg.get('colour_path_line',
                                        PlotConfig.COLOUR_PATH_LINE)
        self.colour_start = cfg.get('colour_start', PlotConfig.COLOUR_START)
        self.colour_goal = cfg.get('colour_goal', PlotConfig.COLOUR_GOAL)
        self.colour_car_state = cfg.get('colour_car_state',
                                        PlotConfig.COLOUR_CAR_STATE)
        self.colour_vertex = cfg.get('colour_vertex', PlotConfig.COLOUR_VERTEX)
        self.colour_segment = cfg.get('colour_segment', PlotConfig.COLOUR_SEGMENT)

        self.factor_turtle_head = cfg.get('factor_turtle_head',
                                          PlotConfig.FACTOR_TURTLE_HEAD)
        self.factor_turtle_line = cfg.get('factor_turtle_line',
                                          PlotConfig.FACTOR_TURTLE_LINE)
        self.factor_parent = cfg.get('factor_parent', PlotConfig.FACTOR_PARENT)
        self.factor_reachable = cfg.get('factor_reachable',
                                        PlotConfig.FACTOR_REACHABLE)
        self.factor_start = cfg.get('factor_start', PlotConfig.FACTOR_START)
        self.factor_goal = cfg.get('factor_goal', PlotConfig.FACTOR_GOAL)
        self.factor_path_node = cfg.get('factor_path_node',
                                        PlotConfig.FACTOR_PATH_NODE)
        self.factor_path_line = cfg.get('factor_path_line',
                                        PlotConfig.FACTOR_PATH_LINE)
        self.factor_car_state = cfg.get('factor_car_state',
                                        PlotConfig.FACTOR_CAR_STATE)
        self.factor_joint = cfg.get('factor_joint', PlotConfig.FACTOR_JOINT)
        self.factor_vertex = cfg.get('factor_vertex', PlotConfig.FACTOR_VERTEX)
        self.factor_edge = cfg.get('factor_edge', PlotConfig.FACTOR_EDGE)
        self.factor_bounds = cfg.get('factor_bounds', PlotConfig.FACTOR_BOUNDS)
        self.factor_segment = cfg.get('factor_segment', PlotConfig.FACTOR_SEGMENT)

        self.bounds_pad = cfg.get('bounds_pad', PlotConfig.BOUNDS_PAD)

        self.plot_parentage = cfg.get('parentage', False)
        self.plot_reachability = cfg.get('reachability', False)
        self.plot_path_nodes = cfg.get('path_nodes', False)
        self.plot_path_line = cfg.get('path_line', True)
        self.normalise_path_line = cfg.get('normalise_path_line', False)
        self.plot_turtle = cfg.get('turtle', False)
        self.plot_vertices = cfg.get('vertices', False)
        self.plot_segments = cfg.get('segments', False)

        self.size_map = cfg.get('size_map', PlotConfig.SIZE_MAP)


class ExecutionConfig:
    BEAMNG_DIR = os.path.join(str(Path.home()), 'Documents/BeamNG.research')
    BEAMNG_EXECUTABLE = 'BeamNG.research.x64.exe'
    BEAMNG_HOME = os.environ['BNG_HOME']

    MODEL_FILE = '/models/self-driving-car-178-2020.h5'
    LEVEL_DIR = 'levels/asfault'
    HOST = 'localhost'
    PORT = 32512
    MAX_SPEED = 'true'

    SPEED_LIMIT = 0

    AI_CONTROLLED = 'true'
    NAVI_GRAPH = 'false'
    ENV_COUNT = 1
    RISK = 1.5
    FAILURE_TIMEOUT_SPM = 0.25
    WAYPOINT_STEP = 75.0
    DIRECTION_AGNOSTIC_BOUNDARY = False
    GOAL_DISTANCE = 10
    MIN_SPEED = 2
    STANDSTILL_THRESHOLD = 120
    DONT_STOP_AT_OBE = False
    OBSERVATION_INTERVAL = 10

    CUSTOM_BEAMNG_TEMPLATE = None


    @staticmethod
    def get_default():
        ret = dict()

        ret['beamng_dir'] = ExecutionConfig.BEAMNG_DIR
        ret['beamng_execuitable'] = ExecutionConfig.BEAMNG_EXECUTABLE
        ret['beamnng_home'] = ExecutionConfig.BEAMNG_HOME
        ret['model_file'] = ExecutionConfig.MODEL_FILE

        ret['level_dir'] = ExecutionConfig.LEVEL_DIR
        ret['host'] = ExecutionConfig.HOST
        ret['port'] = ExecutionConfig.PORT

        ret['max_speed'] = ExecutionConfig.MAX_SPEED
        ret['speed_limit'] = ExecutionConfig.SPEED_LIMIT

        ret['ai_controlled'] = ExecutionConfig.AI_CONTROLLED
        ret['navi_graph'] = ExecutionConfig.NAVI_GRAPH
        ret['env_count'] = ExecutionConfig.ENV_COUNT
        ret['risk'] = ExecutionConfig.RISK
        ret['failure_timeout_spm'] = ExecutionConfig.FAILURE_TIMEOUT_SPM
        ret['waypoint_step'] = ExecutionConfig.WAYPOINT_STEP
        ret['goal_distance'] = ExecutionConfig.GOAL_DISTANCE
        ret['min_speed'] = ExecutionConfig.MIN_SPEED
        ret['standstill_threshold'] = ExecutionConfig.STANDSTILL_THRESHOLD
        ret['direction_agnostic_boundary'] = ExecutionConfig.DIRECTION_AGNOSTIC_BOUNDARY

        ret['dont_stop_at_obe'] = ExecutionConfig.DONT_STOP_AT_OBE
        ret['observation_interval'] = ExecutionConfig.OBSERVATION_INTERVAL

        ret['custom_beamng_template'] = ExecutionConfig.CUSTOM_BEAMNG_TEMPLATE

        return ret

    def __init__(self, path):
        with open(path, 'r') as infile:
            cfg = json.loads(infile.read())

        self.beamng_dir = cfg.get('beamng_dir', ExecutionConfig.BEAMNG_DIR)
        self.beamng_executable = cfg.get('beamng_executable', ExecutionConfig.BEAMNG_EXECUTABLE)

        self.beamng_home = cfg.get('beamng_home', ExecutionConfig.BEAMNG_HOME)
        self.model_file = cfg.get('model_file', ExecutionConfig.MODEL_FILE)


        self.level_dir = cfg.get('level_dir', ExecutionConfig.LEVEL_DIR)
        self.host = cfg.get('host', ExecutionConfig.HOST)
        self.port = cfg.get('port', ExecutionConfig.PORT)
        self.ai_controlled = cfg.get('ai_controlled', ExecutionConfig.AI_CONTROLLED)

        self.max_speed = cfg.get('max_speed', ExecutionConfig.MAX_SPEED)
        self.speed_limit = cfg.get('speed_limit', ExecutionConfig.SPEED_LIMIT)

        self.navi_graph = cfg.get('navi_graph', ExecutionConfig.NAVI_GRAPH)
        self.env_count = cfg.get('env_count', ExecutionConfig.ENV_COUNT)
        self.risk = cfg.get('risk', ExecutionConfig.RISK)
        self.failure_timeout_spm = cfg.get('failure_timeout_spm', ExecutionConfig.FAILURE_TIMEOUT_SPM)
        self.waypoint_step = cfg.get('waypoint_step', ExecutionConfig.WAYPOINT_STEP)
        self.goal_distance = cfg.get('goal_distance', ExecutionConfig.GOAL_DISTANCE)
        self.min_speed = cfg.get('min_speed', ExecutionConfig.MIN_SPEED)
        self.standstill_threshold = cfg.get('standstill_threshold', ExecutionConfig.STANDSTILL_THRESHOLD)
        self.direction_agnostic_boundary = cfg.get('direction_agnostic_boundary', ExecutionConfig.DIRECTION_AGNOSTIC_BOUNDARY)

        self.dont_stop_at_obe = cfg.get('dont_stop_at_obe', ExecutionConfig.DONT_STOP_AT_OBE)
        self.observation_interval = cfg.get('observation_interval', ExecutionConfig.OBSERVATION_INTERVAL)

        self.custom_beamng_template = cfg.get('custom_beamng_template', ExecutionConfig.CUSTOM_BEAMNG_TEMPLATE)

    def get_level_dir(self):
        return os.path.join(self.beamng_dir, self.level_dir)

    def get_user_dir(self):
        return self.beamng_dir


def load_configuration(directory):
    global rg
    global ev
    global pt
    global ex

    rg = AsFaultEnv(directory)
    rg.ensure_directories()

    ev = rg.get_ev_path()
    ev = EvolutionConfig(ev)

    pt = rg.get_pt_path()
    pt = PlotConfig(pt)

    ex = rg.get_ex_path()
    ex = ExecutionConfig(ex)


def write_configuration(path, dict):
    with open(path, 'w') as outfile:
        outfile.write(json.dumps(dict, sort_keys=True, indent=4))


def write_configurations(asfaultv, evolution, plot, execution):
    ev_path = asfaultv.get_ev_path()
    write_configuration(ev_path, evolution)

    pt_path = asfaultv.get_pt_path()
    write_configuration(pt_path, plot)

    ex_path = asfaultv.get_ex_path()
    write_configuration(ex_path, execution)


def get_defaults():
    evolution = EvolutionConfig.get_default()
    plot = PlotConfig.get_default()
    execution = ExecutionConfig.get_default()
    return evolution, plot, execution


def init_configuration(directory):
    global rg
    rg = AsFaultEnv(directory)
    rg.ensure_directories()

    ev_default, pt_default, ex_default = get_defaults()
    write_configurations(rg, ev_default, pt_default, ex_default)
