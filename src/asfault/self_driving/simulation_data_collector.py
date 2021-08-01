from beamngpy import Vehicle, BeamNGpy
from self_driving.beamng_brewer import BeamNGCamera
from self_driving.decal_road import DecalRoad
from self_driving.oob_monitor import OutOfBoundsMonitor
from self_driving.road_polygon import RoadPolygon
from self_driving.simulation_data import SimulationParams, SimulationDataRecords, SimulationData, SimulationDataRecord
from self_driving.vehicle_state_reader import VehicleStateReader


class SimulationDataCollector:

    def __init__(self, vehicle: Vehicle, beamng: BeamNGpy, road: DecalRoad,
                 params: SimulationParams,
                 vehicle_state_reader: VehicleStateReader = None,
                 camera: BeamNGCamera = None,
                 simulation_name: str = None):
        self.vehicle_state_reader = vehicle_state_reader if vehicle_state_reader \
            else VehicleStateReader(vehicle, beamng)
        self.oob_monitor = OutOfBoundsMonitor(RoadPolygon.from_nodes(road.nodes), self.vehicle_state_reader)
        self.beamng: BeamNGpy = beamng
        self.road: DecalRoad = road
        self.params: SimulationParams = params
        self.camera: BeamNGCamera = camera
        self.name = simulation_name
        self.states: SimulationDataRecords = []
        self.simulation_data: SimulationData = SimulationData(simulation_name)
        self.simulation_data.set(self.params, self.road, self.states)
        self.simulation_data.clean()

    def collect_current_data(self, oob_bb=False, wrt="right"):
        """If oob_bb is True, then the out-of-bound (OOB) examples are calculated
        using the bounding box of the car."""
        self.vehicle_state_reader.update_state()
        car_state = self.vehicle_state_reader.get_state()

        is_oob, oob_counter, max_oob_percentage, oob_distance = self.oob_monitor.get_oob_info(oob_bb=oob_bb, wrt=wrt)

        sim_data_record = SimulationDataRecord(**car_state._asdict(),
                                               is_oob=is_oob,
                                               oob_counter=oob_counter,
                                               max_oob_percentage=max_oob_percentage,
                                               oob_distance=oob_distance)
        self.states.append(sim_data_record)

    def get_simulation_data(self) -> SimulationData:
        return self.simulation_data

    def take_car_picture_if_needed(self):
        last_state = self.states[-1]
        if last_state.is_oob:
            self.camera.pose.pos = tuple(last_state.pos[:2]) + (-5,)
            self.camera.pose.rot = (0, 0, -90)
            img_path = self.simulation_data.path_root.joinpath(f'oob_camera_shot{last_state.oob_counter}.jpg')
            img_path.parent.mkdir(parents=True, exist_ok=True)
            if not img_path.exists():
                self.camera.get_rgb_image().save(str(img_path))

    def save(self):
        return self.simulation_data.save()
