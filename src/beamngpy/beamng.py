"""
.. module:: beamng
    :platform: Windows
    :synopsis: Contains the main :py:class:`.BeamNGPy` class used to interface
               with BeamNG.drive.

.. moduleauthor:: Marc Müller <mmueller@beamng.gmbh>

"""

import base64
import logging as log
import mmap
import numpy as np
import os
import queue
import signal
import socket
import subprocess
import sys
import time

from pathlib import Path
from threading import Thread
from time import sleep

import msgpack

from PIL import Image

from .scenario import ScenarioObject

from .beamngcommon import ack
from .beamngcommon import *
from .tig_utils import close_quietly
from .tig_utils import exec_quietly

VERSION = 'v1.12'

BINARIES = [
    'Bin64/BeamNG.research.x64.exe',
    'Bin64/BeamNG.drive.x64.exe',
]


def log_exception(extype, value, trace):
    """
    Hook to log uncaught exceptions to the logging framework. Register this as
    the excepthook with `sys.excepthook = log_exception`.
    """
    log.exception("Uncaught exception: ", exc_info=(extype, value, trace))


def setup_logging(log_file=None):
    """
    Sets up the logging framework to log to the given log_file and to STDOUT.
    If the path to the log_file does not exist, directories for it will be
    created.
    """
    handlers = []
    if log_file:
        if os.path.exists(log_file):
            backup = '{}.1'.format(log_file)
            shutil.move(log_file, backup)
        file_handler = log.FileHandler(log_file, 'w', 'utf-8')
        handlers.append(file_handler)

    term_handler = log.StreamHandler()
    handlers.append(term_handler)
    fmt = '%(asctime)s %(levelname)-8s %(message)s'
    log.basicConfig(handlers=handlers, format=fmt, level=log.DEBUG)

    sys.excepthook = log_exception

    log.info('Started BeamNGpy logging.')


def updating(fun):
    def update_wrapped(*args, **kwargs):
        update_wrapped.__doc__ = fun.__doc__
        if args[0].scenario:
            args[0].update_scenario()
        return fun(*args, **kwargs)
    return update_wrapped


class BeamNGpy:
    """
    The BeamNGpy class is the backbone of communication with the BeamNG
    simulation and offers methods of starting, stopping, and controlling the
    state of the simulator.
    """

    def __init__(self, host, port, home=None, user=None):
        """
        Instantiates a BeamNGpy instance connecting to the simulator on the
        given host and port. The home directory of the simulator can be passed
        to this constructor. If None is given, this class tries to read a
        home path from the ``BNG_HOME`` environment variable.

        Note:
            If no home path is set, this class will not work properly.

        Args:
            host (str): The host to connect to
            port (int): The port to connect to
            home (str): Path to the simulator's home directory.
            user (str): Additional optional user path to set. This path can be
                        used to set where custom files created during
                        executions will be placed if the home folder shall not
                        be touched.
        """
        self.host = host
        self.port = port
        self.next_port = self.port + 1
        self.server = None

        self.home = home
        if not self.home:
            self.home = ENV['BNG_HOME']
        if not self.home:
            raise BNGValueError('No BeamNG home folder given. Either specify '
                                'one in the constructor or define an '
                                'environment variable "BNG_HOME" that '
                                'points to where your copy of BeamNG.* is.')

        self.home = Path(self.home).resolve()
        if user:
            self.user = Path(user).resolve()
        else:
            self.user = None

        self.process = None
        self.skt = None

        self.scenario = None

    def determine_binary(self):
        """
        Tries to find one of the common BeamNG-binaries in the specified home
        path and returns the discovered path as a string.

        Returns:
            Path to the binary as a string.

        Raises:
            BNGError: If no binary could be determined.
        """
        choice = None
        for option in BINARIES:
            binary = self.home / option
            if binary.exists():
                choice = binary
                break

        if not choice:
            raise BNGError('No BeamNG binary found in BeamNG home. Make '
                           'sure any of these exist in the BeamNG home '
                           'folder: %s'.format(','.join(BINARIES)))

        log.debug('Determined BeamNG.* binary to be: %s', choice)
        return str(choice)

    def prepare_call(self):
        """
        Prepares the command line call to execute to start BeamNG.*.
        according to this class' and the global configuration.

        Returns:
            List of shell components ready to be called in the
            :mod:`subprocess` module.
        """
        binary = self.determine_binary()
        call = [
            binary,
            '-console',
            '-rport',
            str(self.port),
            '-rhost',
            str(self.host),
            '-nosteam',
            '-lua',
            "registerCoreModule('{}')".format('util/researchGE'),
        ]

        if self.user:
            call.append('-userpath')
            call.append(str(self.user))

        return call

    def start_beamng(self):
        """
        Spawns a BeamNG.* process and retains a reference to it for later
        termination.
        """
        call = self.prepare_call()
        log.debug('Starting BeamNG process...')
        self.process = subprocess.Popen(call)

    def kill_beamng(self):
        """
        Kills the running BeamNG.* process.
        """
        if not self.process:
            return

        log.debug('Killing BeamNG process...')
        if os.name == "nt":
            with open(os.devnull, 'w') as devnull:
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process.pid)],
                                stdout=devnull, stderr=devnull)
        else:
            os.kill(self.process.pid, signal.SIGTERM)

        self.process = None

    def start_server(self):
        """
        Binds a server socket to the configured host & port and starts
        listening on it.
        """
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen()
        log.info('Started BeamNGpy server on %s:%s', self.host, self.port)

    def send(self, data):
        """
        Helper method for sending data over this instance's socket.

        Args:
            data (dict): The data to send.
        """
        return send_msg(self.skt, data)

    def recv(self):
        """
        Helper method for receiving data over this instance's socket.

        Returns:
            The data received.
        """
        return recv_msg(self.skt)

    def open(self, launch=True):
        """
        Starts a BeamNG.* process, opens a server socket, and waits for the
        spawned BeamNG.* process to connect. This method blocks until the
        process started and is ready.

        Args:
            launch (bool): Whether to launch a new process or connect to a
                           running one on the configured host/port. Defaults to
                           True.
        """
        log.info('Opening BeamNPy instance...')
        self.start_server()
        if launch:
            self.start_beamng()

        self.server.settimeout(300)
        self.skt, addr = self.server.accept()
        self.skt.settimeout(300)

        log.debug('Connection established. Awaiting "hello"...')
        hello = self.recv()
        assert hello['type'] == 'Hello'
        if hello['version'] != VERSION:
            print('BeamNGpy and BeamNG.* version mismatch: '
                  'BeamNGpy {}, BeamNG.* {}'.format(VERSION, hello['version']))
            print('Make sure both this library and BeamNG.* are up to date.')
            print('Operation will proceed, but some features might not work.')

        log.info('Started BeamNGpy communicating on %s', addr)
        return self

    def close(self):
        """
        Kills the BeamNG.* process and closes the server.
        """
        log.info('Closing BeamNGpy instance...')
        if self.scenario:
            self.scenario.close()
            self.scenario = None

        close_quietly(self.skt)
        close_quietly(self.server)
        self.kill_beamng()

    def hide_hud(self):
        """
        Hides the HUD in the simulator.
        """
        data = dict(type='HideHUD')
        self.send(data)

    def show_hud(self):
        """
        Shows the HUD in the simulator.
        """
        data = dict(type='ShowHUD')
        self.send(data)

    def connect_vehicle(self, vehicle, port=None):
        """
        Creates a server socket for the given vehicle and sends a connection
        request for it to the simulation. This method does not wait for the
        connection to be established but rather returns the respective server
        socket to the caller.

        Args:
            vehicle (:class:`.Vehicle`): The vehicle instance to be connected.
            port (int): Optional. The port the vehicle should be connecting
                        over.

        Returns:
            The server socket created and waiting for a conection.
        """
        flags = vehicle.get_engine_flags()
        self.set_engine_flags(flags)

        if not port:
            port = self.next_port
            self.next_port += 1

        vehicle_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        vehicle_server.bind((self.host, port))
        vehicle_server.listen()
        log.debug('Starting vehicle server for %s on: %s:%s',
                  vehicle, self.host, port)

        connection_msg = {'type': 'VehicleConnection'}
        connection_msg['vid'] = vehicle.vid
        connection_msg['host'] = self.host
        connection_msg['port'] = port

        self.send(connection_msg)

        vehicle.connect(self, vehicle_server, port)
        vehicle.update_vehicle()
        return vehicle_server

    def setup_vehicles(self, scenario):
        """
        Goes over the current scenario's vehicles and establishes a connection
        between their vehicle instances and the vehicles in simulation. Engine
        flags required by the vehicles' sensor setups are sent and connect-
        hooks of the respective sensors called upon connection. This method
        blocks until all vehicles are fully connected.

        Args:
            scenario (:class:`.Scenario`): Calls functions to set up scenario
                                           objects after it has been loaded.
        """
        vehicles = scenario.vehicles
        for vehicle in vehicles.keys():
            self.connect_vehicle(vehicle)

    def load_scenario(self, scenario):
        """
        Loads the given scenario in the simulation and returns once loading
        is finished.

        Args:
            scenario (:class:`.Scenario`): The scenario to load.
        """
        info_path = scenario.get_info_path()
        info_path = info_path.replace(str(self.home), '')
        info_path = info_path.replace(str(self.user), '')
        info_path = info_path[1:]
        info_path = info_path.replace('\\', '/')
        data = {'type': 'LoadScenario', 'path': info_path}
        self.send(data)
        resp = self.recv()
        assert resp['type'] == 'MapLoaded'
        self.setup_vehicles(scenario)
        flags = scenario.get_engine_flags()
        self.set_engine_flags(flags)
        self.scenario = scenario
        self.scenario.connect(self)

    @ack('SetEngineFlags')
    def set_engine_flags(self, flags):
        """
        Sets flags in the simulation engine. Flags are given as key/value pairs
        of strings and booleans, where each string specifies a flag and the
        boolean the state to set. Possible flags are:

         * ``annotations``: Whether pixel-wise annotation should be enabled.
         * ``lidar``: Whether Lidar rendering should be enabled.

        """
        flags = dict(type='EngineFlags', flags=flags)
        self.send(flags)

    @ack('OpenedShmem')
    def open_shmem(self, name, size):
        """
        Tells the simulator to open a shared memory handle with the given
        amount of bytes.

        Args:
            name (str): The name of the shared memory to open.
            size (int): The size to map in bytes.
        """
        data = dict(type='OpenShmem', name=name, size=size)
        self.send(data)

    @ack('ClosedShmem')
    def close_shmem(self, name):
        """
        Tells the simulator to close a previously-opened shared memory handle.

        Args:
            name (str): The name of the shared memory space to close.
        """
        data = dict(type='CloseShmem', name=name)
        self.send(data)


    @ack('OpenedLidar')
    def open_lidar(self, name, vehicle, shmem, shmem_size, offset=(0, 0, 0),
                   direction=(0, -1, 0), vres=64, vangle=26.9, rps=2200000,
                   hz=20, angle=360, max_dist=120, visualized=True):
        """
        Opens a Lidar sensor instance in the simulator with the given
        parameters writing its data to the given shared memory space. The Lidar
        instance has to be assigned a unique name that is later used for
        closing.

        Args:
            name (str): The name of the Lidar instance to open. Has to be
                        unique relative to other Lidars currently opened.
            vehicle (:class:`.Vehicle`): The vehicle this Lidar is attached to.
            shmem (str): The handle of the shared memory space used to exchange
                         data.
            shmem_size (int): Size of the shared memory space that has been
                              allocated for exchange.
            offset (tuple): (X, Y, Z) coordinate triplet specifying the
                            position of the sensor relative to the vehicle's.
            direction (tuple): (X, Y, Z) coordinate triple specifying the
                               direction the Lidar is pointing towards.
            vres (int): Vertical resolution, i.e. how many lines are sampled
                        vertically.
            vangle (float): The vertical angle, i.e. how many degrees up and
                            down points are scattered.
            rps (int): The rays per second shot by the sensor.
            hz (int): The refresh rate of the sensor in Hz
            angle (float): The horizontal degrees covered, i.e. 360 degrees
                           covers the entire surroundings of the vehicle.
            max_dist (float): Maximum distance of points. Any dot farther away
                              will not show up in the sample.
            visualized (bool): Whether or not to render the Lidar sensor's
                              points in the simulator.
        """
        data = dict(type='OpenLidar')
        data['name'] = name
        data['shmem'] = shmem
        data['size'] = shmem_size
        data['vid'] = vehicle.vid
        data['offset'] = offset
        data['direction'] = direction
        data['vRes'] = vres
        data['vAngle'] = vangle
        data['rps'] = rps
        data['hz'] = hz
        data['angle'] = angle
        data['maxDist'] = max_dist
        data['visualized'] = visualized
        self.send(data)

    @ack('ClosedLidar')
    def close_lidar(self, name):
        """
        Closes the Lidar instance of the given name in the simulator.

        Args:
            name (str): The name of the Lidar instance to close.
        """
        data = dict(type='CloseLidar')
        data['name'] = name
        self.send(data)

    @ack('Teleported')
    def teleport_vehicle(self, vehicle, pos, rot=None):
        """
        Teleports the given vehicle to the given position with the given
        rotation.

        Args:
            vehicle (:class:`.Vehicle`): The vehicle to teleport.
            pos (tuple): The target position as an (x,y,z) tuple containing
                         world-space coordinates.
            rot (tuple): Optional tuple specifying rotations around the (x,y,z)
                         axes in degrees.

        Notes:
            In the current implementation, if both ``pos`` and ``rot`` are
            specified, the vehicle will be repaired to its initial state during
            teleport.
        """
        data = dict(type='Teleport')
        data['vehicle'] = vehicle.vid
        data['pos'] = pos
        if rot:
            data['rot'] = [np.radians(r) for r in rot]
        self.send(data)

    @ack('ScenarioStarted')
    def start_scenario(self):
        """
        Starts the scenario; equivalent to clicking the "Start" button in the
        game after loading a scenario. This method blocks until the countdown
        to the scenario's start has finished.
        """
        data = dict(type="StartScenario")
        self.send(data)

    @ack('ScenarioRestarted')
    def restart_scenario(self):
        """
        Restarts a running scenario.
        """
        if not self.scenario:
            raise BNGError('Need to have a scenario loaded to restart it.')

        self.scenario.restart()

        data = dict(type='RestartScenario')
        self.send(data)

    @ack('ScenarioStopped')
    def stop_scenario(self):
        """
        Stops a running scenario and returns to the main menu.
        """
        if not self.scenario:
            raise BNGError('Need to have a scenario loaded to stop it.')

        self.scenario.close()
        self.scenario = None

        data = dict(type='StopScenario')
        self.send(data)

    @ack('SetPhysicsDeterministic')
    def set_deterministic(self):
        """
        Sets the simulator to run in deterministic mode. For this to function
        properly, an amount of steps per second needs to have been specified
        in the simulator's settings, or through
        :meth:`~.BeamnGpy.set_steps_per_second`.
        """
        data = dict(type='SetPhysicsDeterministic')
        self.send(data)

    @ack('SetPhysicsNonDeterministic')
    def set_nondeterministic(self):
        """
        Disables the deterministic mode of the simulator. Any steps per second
        setting is retained.
        """
        data = dict(type='SetPhysicsNonDeterministic')
        self.send(data)

    @ack('SetFPSLimit')
    def set_steps_per_second(self, sps):
        """
        Specifies the temporal resolution of the simulation. The setting can be
        understood to determine into how many steps the simulation divides one
        second of simulation. A setting of two, for example, would mean one
        second is simulated in two steps. Conversely, to simulate one second,
        one needs to advance the simulation two steps.

        Args:
            sps (int): The steps per second to set.
        """
        data = dict(type='FPSLimit', fps=sps)
        self.send(data)

    @ack('RemovedFPSLimit')
    def remove_step_limit(self):
        """
        Removes the steps-per-second setting, making the simulation run at
        undefined time slices.
        """
        data = dict(type='RemoveFPSLimit')
        self.send(data)

    def step(self, count, wait=True):
        """
        Advances the simulation the given amount of steps, assuming it is
        currently paused. If the wait flag is set, this method blocks until
        the simulator has finished simulating the desired amount of steps. If
        not, this method resumes immediatly. This can be used to queue commands
        that should be executed right after the steps have been simulated.

        Args:
            count (int): The amount of steps to simulate.
            wait (bool): Optional. Whether to wait for the steps to be
                         simulated. Defaults to True.

        Raises:
            BNGError: If the wait flag is set but the simulator doesn't respond
                      appropriately.
        """
        data = dict(type='Step', count=count)
        data['ack'] = wait
        self.send(data)
        if wait:
            resp = self.recv()
            if resp['type'] != 'Stepped':
                raise BNGError('Wrong ACK: {} != {}'.format('Stepped',
                                                            resp['type']))

    @ack('Paused')
    def pause(self):
        """
        Sends a pause request to BeamNG.*, blocking until the simulation is
        paused.
        """
        data = dict(type='Pause')
        self.send(data)

    @ack('Resumed')
    def resume(self):
        """
        Sends a resume request to BeamNG.*, blocking until the simulation
        is resumed.
        """
        data = dict(type='Resume')
        self.send(data)

    def poll_sensors(self, vehicle):
        """
        Retrieves sensor values for the sensors attached to the given vehicle.
        This method correctly splits requests meant for the game engine and
        requests meant for the vehicle, sending them to their supposed
        destinations and waiting for results from them. Results from either are
        merged into one dictionary for ease of use. The received data is
        decoded by each sensor and returned, but also stored in the vehicle's
        sensor cache to avoid repeated requests.

        Args:
            vehicle (:class:`.Vehicle`): The vehicle whose sensors are polled.

        Returns:
            The decoded sensor data from both engine and vehicle as one
            dictionary having a key-value pair for each sensor's name and the
            data received for it.
        """
        engine_reqs, vehicle_reqs = vehicle.encode_sensor_requests()
        sensor_data = dict()

        if engine_reqs['sensors']:
            start = time.time()
            self.send(engine_reqs)
            response = self.recv()
            assert response['type'] == 'SensorData'
            sensor_data.update(response['data'])

        if vehicle_reqs['sensors']:
            response = vehicle.poll_sensors(vehicle_reqs)
            sensor_data.update(response)
        else:
            vehicle.update_vehicle()

        result = vehicle.decode_sensor_response(sensor_data)
        vehicle.sensor_cache = result
        return result

    def render_cameras(self):
        """
        Renders all cameras associated with the loaded scenario. These cameras
        work exactly like the ones attached to vehicles as sensors, except
        scenario cameras do not follow the vehicle they are attached to and can
        be used to get a view from the perspective of something like a
        surveillance camera, for example.

        A scenario needs to be loaded for this method to work.

        Returns:
            The rendered data for all cameras in the loaded scenario as a
            dict mapping camera name to render results.
        """
        if not self.scenario:
            raise BNGError('Need to be in a started scenario to render its '
                           'cameras.')

        engine_reqs = self.scenario.encode_requests()
        self.send(engine_reqs)
        response = self.recv()
        assert response['type'] == 'SensorData'
        camera_data = response['data']
        result = self.scenario.decode_frames(camera_data)
        return result

    def get_roads(self):
        """
        Retrieves the vertex data of all DecalRoads in the current scenario.
        The vertex data of a DecalRoad is formatted as point triples, where
        each triplet represents the left, centre, and right points of the edges
        that make up a DecalRoad.

        Returns:
            A dict mapping DecalRoad IDs to lists of point triples.
        """
        if not self.scenario:
            raise BNGError('Need to be in a started scenario to get its '
                           'DecalRoad data.')

        data = dict(type='GetDecalRoadData')
        self.send(data)
        response = self.recv()
        assert response['type'] == 'DecalRoadData'
        return response['data']

    def get_road_edges(self, road):
        """
        Retrieves the edges of the road with the given name and returns them
        as a list of point triplets. Roads are defined by a series of lines
        that specify the leftmost, center, and rightmost point in the road.
        These lines go horizontally across the road and the series of leftmost
        points make up the left edge of the road, the series of rightmost
        points make up the right edge of the road, and the series of center
        points the middle line of the road.

        Args:
            road (str): Name of the road to get edges from.

        Returns:
            The road edges as a list of (left, center, right) point triplets.
            Each point is an (X, Y, Z) coordinate triplet.
        """
        data = dict(type='GetDecalRoadEdges')
        data['road'] = road
        self.send(data)
        response = self.recv()
        assert response['type'] == 'DecalRoadEdges'
        return response['edges']

    def get_gamestate(self):
        """
        Retrieves the current game state of the simulator. The game state is
        returned as a dictionary containing a ``state`` entry that is either:

            * ``scenario`` when a scenario is loaded
            * ``menu`` otherwise

        If a scenario is loaded, the resulting dictionary also contains a
        ``scenario_state`` entry whose value is ``pre-running`` if the scenario
        is currently at the start screen or ``running`` otherwise.

        Returns:
            The game state as a dictionary as described above.
        """
        data = dict(type='GameStateRequest')
        self.send(data)
        resp = self.recv()
        assert resp['type'] == 'GameState'
        return resp

    @ack('TimeOfDayChanged')
    def set_tod(self, tod):
        """
        Sets the current time of day. The time of day value is given as a float
        between 0 and 1. How this value affects the lighting of the scene is
        dependant on the map's TimeOfDay object.

        Args:
            tod (float): Time of day beteen 0 and 1.
        """
        data = dict(type='TimeOfDayChange')
        data['tod'] = tod
        self.send(data)

    @ack('WeatherPresetChanged')
    def set_weather_preset(self, preset, time=1):
        """
        Triggers a change to a different weather preset. Weather presets affect
        multiple settings at once (time of day, wind speed, cloud coverage,
        etc.) and need to have been defined first. Example json objects
        defining weather presets can be found in BeamNG.research's
        ``art/weather/defaults.json`` file.

        Args:
            preset (str): The name of the preset to switch to. Needs to be
                          defined already within the simulation.
            time (float): Time in seconds the transition from the current
                          settings to the preset's should take.
        """
        data = dict(type='SetWeatherPreset')
        data['preset'] = preset
        data['time'] = time
        self.send(data)

    def await_vehicle_spawn(self, vid):
        """
        Waits for the vehicle with the given name to spawn and returns once it
        has.

        Args:
            vid (str): The name of the  vehicle to wait for.
        """
        data = dict(type='WaitForSpawn')
        data['name'] = vid
        self.send(data)
        resp = self.recv()
        assert resp['type'] == 'VehicleSpawned'
        assert resp['name'] == vid

    def update_scenario(self):
        """
        Updates the :attr:`.Vehicle.state` field of each vehicle in the
        currently running scenario.
        """
        if not self.scenario:
            raise BNGError('Need to have a senario loaded to update it.')

        data = dict(type='UpdateScenario')
        data['vehicles'] = list()
        for vehicle in self.scenario.vehicles.keys():
            data['vehicles'].append(vehicle.vid)
        self.send(data)
        resp = self.recv()
        assert resp['type'] == 'ScenarioUpdate'
        for name, vehicle_state in resp['vehicles'].items():
            vehicle = self.scenario.get_vehicle(name)
            if vehicle:
                vehicle.state = vehicle_state

    @ack('GuiMessageDisplayed')
    def display_gui_message(self, msg):
        """
        Displays a toast message in the user interface of the simulator.

        Args:
            msg (str): The message to display.
        """
        data = dict(type='DisplayGuiMessage')
        data['message'] = msg
        self.send(data)

    @ack('VehicleSwitched')
    def switch_vehicle(self, vehicle):
        """
        Switches to the given :class:`.Vehicle`. This means that the
        simulator's main camera, inputs by the user, and so on will all focus
        on that vehicle from now on.

        Args:
            vehicle (:class:`.Vehicle`): The target vehicle.
        """
        data = dict(type='SwitchVehicle')
        data['vid'] = vehicle.vid
        self.send(data)

    @ack('FreeCameraSet')
    def set_free_camera(self, pos, direction):
        """
        Sets the position and direction of the free camera. The free camera is
        one that does not follow any particular vehicle, but can instead be
        put at any spot and any position on the map.

        Args:
            pos (tuple): The position of the camera as a (x, y, z) triplet.
            direction (tuple): The directional vector of the camera as a
                               (x, y, z) triplet.
        """
        data = dict(type='SetFreeCamera')
        data['pos'] = pos
        data['dir'] = direction
        self.send(data)

    @ack('ParticlesSet')
    def set_particles_enabled(self, enabled):
        """
        En-/disabled visual particle emmission.

        Args:
            enabled (bool): Whether or not to en- or disabled effects.
        """
        data = dict(type='ParticlesEnabled')
        data['enabled'] = enabled
        self.send(data)

    @ack('PartsAnnotated')
    def annotate_parts(self, vehicle):
        """
        Triggers per-part annotation for the given :class:`.Vehicle`.

        Args:
            vehicle (:class:`.Vehicle`): The vehicle to annotate.
        """
        data = dict(type='AnnotateParts')
        data['vid'] = vehicle.vid
        self.send(data)

    def get_scenario_name(self):
        """
        Retrieves the name of the currently-loaded scenario in the simulator.

        Returns:
            The name of the loaded scenario as a string.
        """
        data = dict(type='GetScenarioName')
        self.send(data)
        resp = self.recv()
        assert resp['type'] == 'ScenarioName'
        return resp['name']

    def spawn_vehicle(self, vehicle, pos, rot, cling=True):
        """
        Spawns the given :class:`.Vehicle` instance in the simulator. This
        method is meant for spawning vehicles *during the simulation*. Vehicles
        that are known to be required before running the simulation should be
        added during scenario creation instead.

        Args:
            vehicle (:class:`.Vehicle`): The vehicle to be spawned.
            pos (tuple): Where to spawn the vehicle as a (x, y, z) triplet.
            rot (tuple): The rotation of the vehicle as a triplet of Euler
                         angles.
            cling (bool): If set, the z-coordinate of the vehicle's position
                          will be set to the ground level at the given
                          position to avoid spawning the vehicle below ground
                          or in the air.
        """
        data = dict(type='SpawnVehicle', cling=cling)
        data['name'] = vehicle.vid
        data['model'] = vehicle.options['model']
        data['pos'] = pos
        data['rot'] = rot
        data.update(vehicle.options)
        self.send(data)
        resp = self.recv()
        self.connect_vehicle(vehicle)
        assert resp['type'] == 'VehicleSpawned'

    def despawn_vehicle(self, vehicle):
        """
        Despawns the given :class:`.Vehicle` from the simulation.

        Args:
            vehicle (:class:`.Vehicle`): The vehicle to despawn.
        """
        vehicle.disconnect()
        data = dict(type='DespawnVehicle')
        data['vid'] = vehicle.vid
        self.send(data)
        resp = self.recv()
        assert resp['type'] == 'VehicleDespawned'

    def find_objects_class(self, clazz):
        """
        Scans the current environment in the simulator for objects of a
        certain class and returns them as a list of :class:`.ScenarioObject`.

        What kind of classes correspond to what kind of objects is described
        in the BeamNG.drive documentation.

        Args:
            clazz (str): The class name of objects to find.

        Returns:
            Found objects as a list.
        """
        data = dict(type='FindObjectsClass')
        data['class'] = clazz
        self.send(data)
        resp = self.recv()
        ret = list()
        for obj in resp['objects']:
            sobj = ScenarioObject(obj['id'], obj['name'], obj['type'],
                                  tuple(obj['position']),
                                  tuple(obj['rotation']),
                                  tuple(obj['scale']),
                                  **obj['options'])
            ret.append(sobj)

        return ret

    @ack('CreatedCylinder')
    def create_cylinder(self, radius, height, pos, rot,
                        material=None, name=None):
        """
        Creates a procedurally generated cylinder mesh with the given
        radius and height at the given position and rotation. The material
        can optionally be specified and a name can be assigned for later
        identification.

        Args:
            radius (float): The radius of the cylinder's base circle.
            height (float): The between top and bottom circles of the
                            cylinder.
            pos (tuple): (X, Y, Z) coordinate triplet specifying the cylinder's
                         position.
            rot (tuple): Triplet of Euler angles specifying rotations around
                         the (X, Y, Z) axes.
            material (str): Optional material name to use as a texture for the
                            mesh.
            name (str): Optional name for the mesh.
        """
        data = dict(type='CreateCylinder')
        data['radius'] = radius
        data['height'] = height
        data['pos'] = pos
        data['rot'] = rot
        data['name'] = name
        data['material'] = material
        self.send(data)

    @ack('CreatedBump')
    def create_bump(self, width, length, height, upper_length, upper_width,
                    pos, rot, material=None, name=None):
        """
        Creates a procedurally generated bump with the given properties at the
        given position and rotation. The material can optionally be specified
        and a name can be assigned for later identification.

        Args:
            width (float): The width of the bump, i.e. its size between left
                           and right edges.
            length (float): The length of the bump, i.e. the distances from
                            up and downward slopes.
            height (float): The height of the tip.
            upper_length (float): The length of the tip.
            upper_width (float): The width of the tip.
            pos (tuple): (X, Y, Z) coordinate triplet specifying the cylinder's
                         position.
            rot (tuple): Triplet of Euler angles specifying rotations around
                         the (X, Y, Z) axes.
            material (str): Optional material name to use as a texture for the
                            mesh.
            name (str): Optional name for the mesh.
        """
        data = dict(type='CreateBump')
        data['width'] = width
        data['length'] = length
        data['height'] = height
        data['upperLength'] = upper_length
        data['upperWidth'] = upper_width
        data['pos'] = pos
        data['rot'] = rot
        data['name'] = name
        data['material'] = material
        self.send(data)

    @ack('CreatedCone')
    def create_cone(self, radius, height, pos, rot, material=None, name=None):
        """
        Creates a procedurally generated cone with the given properties at the
        given position and rotation. The material can optionally be specified
        and a name can be assigned for later identification.

        Args:
            radius (float): Radius of the base circle.
            height (float): Distance of the tip to the base circle.
            pos (tuple): (X, Y, Z) coordinate triplet specifying the cylinder's
                         position.
            rot (tuple): Triplet of Euler angles specifying rotations around
                         the (X, Y, Z) axes.
            material (str): Optional material name to use as a texture for the
                            mesh.
            name (str): Optional name for the mesh.
        """
        data = dict(type='CreateCone')
        data['radius'] = radius
        data['height'] = height
        data['material'] = material
        data['name'] = name
        data['pos'] = pos
        data['rot'] = rot
        self.send(data)

    @ack('CreatedCube')
    def create_cube(self, size, pos, rot, material=None, name=None):
        """
        Creates a procedurally generated cube with the given properties at the
        given position and rotation. The material can optionally be specified
        and a name can be assigned for later identification.

        Args:
            size (tuple): A triplet specifying the (length, width, height) of
                          the cuboid.
            pos (tuple): (X, Y, Z) coordinate triplet specifying the cylinder's
                         position.
            rot (tuple): Triplet of Euler angles specifying rotations around
                         the (X, Y, Z) axes.
            material (str): Optional material name to use as a texture for the
                            mesh.
            name (str): Optional name for the mesh.
        """
        data = dict(type='CreateCube')
        data['size'] = size
        data['pos'] = pos
        data['rot'] = rot
        data['material'] = material
        data['name'] = name
        self.send(data)

    @ack('CreatedRing')
    def create_ring(self, radius, thickness, pos, rot,
                    material=None, name=None):
        """
        Creates a procedurally generated ring with the given properties at the
        given position and rotation. The material can optionally be specified
        and a name can be assigned for later identification.

        Args:
            radius (float): Radius of the circle encompassing the ring.
            thickness (float): Thickness of the rim.
            pos (tuple): (X, Y, Z) coordinate triplet specifying the cylinder's
                         position.
            rot (tuple): Triplet of Euler angles specifying rotations around
                         the (X, Y, Z) axes.
            material (str): Optional material name to use as a texture for the
                            mesh.
            name (str): Optional name for the mesh.
        """
        data = dict(type='CreateRing')
        data['radius'] = radius
        data['thickness'] = thickness
        data['pos'] = pos
        data['rot'] = rot
        data['material'] = material
        data['name'] = name
        self.send(data)

    def get_vehicle_bbox(self, vehicle):
        data = dict(type='GetBBoxCorners')
        data['vid'] = vehicle.vid
        self.send(data)
        resp = self.recv()
        assert resp['type'] == 'BBoxCorners'
        points = resp['points']
        bbox = {
            'near_bottom_left': points[3],
            'near_bottom_right': points[0],
            'near_top_left': points[2],
            'near_top_right': points[1],
            'far_bottom_left': points[7],
            'far_bottom_right': points[4],
            'far_top_left': points[6],
            'far_top_right': points[5],
        }
        return bbox

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
