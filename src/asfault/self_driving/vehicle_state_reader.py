from collections import namedtuple
import numpy as np
from beamngpy import Vehicle, BeamNGpy
from beamngpy.sensors import GForces, Electrics, Damage, Timer, Sensor
from typing import List, Tuple

VehicleStateProperties = ['timer', 'damage', 'pos', 'dir', 'vel', 'gforces', 'gforces2', 'steering', 'steering_input',
                          'brake', 'brake_input', 'throttle', 'throttle_input', 'throttleFactor', 'engineThrottle',
                          'wheelspeed', 'vel_kmh']

VehicleState = namedtuple('VehicleState', VehicleStateProperties)


class VehicleStateReader:
    def __init__(self, vehicle: Vehicle, beamng: BeamNGpy, additional_sensors: List[Tuple[str, Sensor]] = None):
        self.vehicle = vehicle
        self.beamng = beamng
        self.state: VehicleState = None
        self.vehicle_state = {}

        gforces = GForces()
        electrics = Electrics()
        damage = Damage()
        timer = Timer()

        self.vehicle.attach_sensor('gforces', gforces)
        self.vehicle.attach_sensor('electrics', electrics)
        self.vehicle.attach_sensor('damage', damage)
        self.vehicle.attach_sensor('timer', timer)

        if additional_sensors:
            for (name, sensor) in additional_sensors:
                self.vehicle.attach_sensor(name, sensor)

    def get_state(self) -> VehicleState:
        return self.state

    def get_vehicle_bbox(self) -> dict:
        return self.vehicle.get_bbox()

    def update_state(self):
        sensors = self.beamng.poll_sensors(self.vehicle)
        self.sensors = sensors

        self.vehicle.update_vehicle()
        st = self.vehicle.state

        ele = sensors['electrics']['values']
        gforces = sensors['gforces']

        vel = tuple(st['vel'])
        self.state = VehicleState(timer=sensors['timer']['time']
                                  , damage=sensors['damage']['damage']
                                  , pos=tuple(st['pos'])
                                  , dir=tuple(st['dir'])
                                  , vel=vel
                                  , gforces=(gforces['gx'], gforces['gy'], gforces['gz'])
                                  , gforces2=(gforces['gx2'], gforces['gy2'], gforces['gz2'])
                                  , steering=ele.get('steering', None)
                                  , steering_input=ele.get('steering_input', None)
                                  , brake=ele.get('brake', None)
                                  , brake_input=ele.get('brake_input', None)
                                  , throttle=ele.get('throttle', None)
                                  , throttle_input=ele.get('throttle_input', None)
                                  , throttleFactor=ele.get('throttleFactor', None)
                                  , engineThrottle=ele.get('engineThrottle', None)
                                  , wheelspeed=ele.get('wheelspeed', None)
                                  , vel_kmh=int(round(np.linalg.norm(vel) * 3.6)))
