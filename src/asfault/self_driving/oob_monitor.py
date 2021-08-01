from typing import Tuple

from shapely.geometry import Point, Polygon

from self_driving.road_polygon import RoadPolygon
from self_driving.vehicle_state_reader import VehicleStateReader


class OutOfBoundsMonitor:
    def __init__(self, road_polygon: RoadPolygon, vehicle_state_reader: VehicleStateReader):
        assert isinstance(vehicle_state_reader, VehicleStateReader)
        assert isinstance(road_polygon, RoadPolygon)
        self.road_polygon = road_polygon
        self.vehicle_state_reader = vehicle_state_reader
        self.oob_counter = 0
        self.last_is_oob = False
        self.last_max_oob_percentage = 0

    def get_oob_info(self, wrt="right", oob_bb=False, tolerance=0.05) -> Tuple[bool, int, float]:
        if oob_bb:
            is_oob = self.is_oob_bb(tolerance=tolerance, wrt=wrt)
            self.update_oob_percentage(is_oob)
        else:
            is_oob = self.is_oob(wrt=wrt)

        self.update_oob_counter(is_oob)

        last_max_oob_percentage = self.last_max_oob_percentage if oob_bb else float("nan")
        oob_distance = float("nan") if oob_bb else self.oob_distance( wrt=wrt)

        return is_oob, self.oob_counter, last_max_oob_percentage, oob_distance

    def update_oob_counter(self, is_oob: bool):
        """Update the OOB counter only when is_oob is True but self.last_is_oob is False."""
        if not self.last_is_oob and is_oob:
            self.oob_counter += 1
            self.last_is_oob = True
        elif self.last_is_oob and not is_oob:
            self.last_is_oob = False

    def update_oob_percentage(self, is_oob: bool):
        if not self.last_is_oob and is_oob:
            self.last_max_oob_percentage = self.oob_percentage()
        elif self.last_is_oob and is_oob:
            self.last_max_oob_percentage = max(self.last_max_oob_percentage, self.oob_percentage())

    def oob_percentage(self, wrt="right") -> float:
        """Returns the percentage of the bounding box of the car with respect to
        one of the lanes of the road or the road itself (depending on the value of wrt)."""
        car_bbox_polygon = self._get_car_bbox_polygon()
        if wrt == "right":
            intersection = car_bbox_polygon.intersection(self.road_polygon.right_polygon)
        elif wrt == "left":
            intersection = car_bbox_polygon.intersection(self.road_polygon.left_polygon)
        else:
            intersection = car_bbox_polygon.intersection(self.road_polygon.polygon)
        return 1 - intersection.area / car_bbox_polygon.area

    def is_oob_bb(self, tolerance=0.05, wrt="right") -> bool:
        """Returns true if the bounding box of the car is more than tolerance
        percentage outside of the road."""
        return self.oob_percentage(wrt=wrt) > tolerance

    def oob_distance(self,  wrt="right") -> float:
        """Returns the difference between the width of a lane and
        the distance between the car and the center of the road."""
        car_point = Point(self.vehicle_state_reader.get_state().pos)
        divisor = 4.0
        if wrt == "right":
            distance = self.road_polygon.right_polyline.distance(car_point)
        elif wrt == "left":
            distance = self.road_polygon.left_polyline.distance(car_point)
        else:
            distance = self.road_polygon.polyline.distance(car_point)
            divisor = 2.0
        difference = self.road_polygon.road_width / divisor - distance
        return difference

    def is_oob(self, wrt="right") -> bool:
        """Returns true if the car is an out-of-bound (OOB).

        The OOB can be calculated with respect to the left or right lanes,
        or with respect to the whole road.

        The car position is represented by the center of mass of the car.
        If you want to calculate the OOBs using the bounding box of the car,
        call self.is_oob_bb."""
        car_point = Point(self.vehicle_state_reader.get_state().pos)
        if wrt == "right":
            return not self.road_polygon.right_polygon.contains(car_point)
        elif wrt == "left":
            return not self.road_polygon.left_polygon.contains(car_point)
        else:
            return not self.road_polygon.polygon.contains(car_point)

    def _get_car_bbox_polygon(self) -> Polygon:
        car_bbox = self.vehicle_state_reader.get_vehicle_bbox()

        # x coordinates of the bounding box of the car.
        boundary_x = [
            car_bbox['near_bottom_left'][0],
            car_bbox['near_bottom_right'][0],
            car_bbox['far_bottom_right'][0],
            car_bbox['far_bottom_left'][0],
            car_bbox['near_bottom_left'][0],
        ]

        # y coordinates of the bounding box of the car.
        boundary_y = [
            car_bbox['near_bottom_left'][1],
            car_bbox['near_bottom_right'][1],
            car_bbox['far_bottom_right'][1],
            car_bbox['far_bottom_left'][1],
            car_bbox['near_bottom_left'][1],
        ]

        return Polygon(zip(boundary_x, boundary_y))



class OutOfBoundsMonitorAsFault:
    def __init__(self, road_polygon: RoadPolygon, car_pose, car_bbox):
        self.road_polygon = road_polygon
        self.car_pose = car_pose
        self.car_bbox = car_bbox
        self.oob_counter = 0
        self.last_is_oob = False
        self.last_max_oob_percentage = 0

    def get_oob_info(self, wrt="right", oob_bb=False, tolerance=0.05) -> Tuple[bool, int, float]:
        if oob_bb:
            is_oob = self.is_oob_bb(tolerance=tolerance, wrt=wrt)
            self.update_oob_percentage(is_oob)
        else:
            is_oob = self.is_oob(wrt=wrt)

        self.update_oob_counter(is_oob)

        last_max_oob_percentage = self.last_max_oob_percentage if oob_bb else float("nan")
        oob_distance = float("nan") if oob_bb else self.oob_distance( wrt=wrt)

        return is_oob, self.oob_counter, last_max_oob_percentage, oob_distance

    def update_oob_counter(self, is_oob: bool):
        """Update the OOB counter only when is_oob is True but self.last_is_oob is False."""
        if not self.last_is_oob and is_oob:
            self.oob_counter += 1
            self.last_is_oob = True
        elif self.last_is_oob and not is_oob:
            self.last_is_oob = False

    def update_oob_percentage(self, is_oob: bool):
        if not self.last_is_oob and is_oob:
            self.last_max_oob_percentage = self.oob_percentage()
        elif self.last_is_oob and is_oob:
            self.last_max_oob_percentage = max(self.last_max_oob_percentage, self.oob_percentage())

    def oob_percentage(self, wrt="right") -> float:
        """Returns the percentage of the bounding box of the car with respect to
        one of the lanes of the road or the road itself (depending on the value of wrt)."""
        car_bbox_polygon = self._get_car_bbox_polygon()
        if wrt == "right":
            intersection = car_bbox_polygon.intersection(self.road_polygon.right_polygon)
        elif wrt == "left":
            intersection = car_bbox_polygon.intersection(self.road_polygon.left_polygon)
        else:
            intersection = car_bbox_polygon.intersection(self.road_polygon.polygon)
        return 1 - intersection.area / car_bbox_polygon.area

    def is_oob_bb(self, tolerance=0.05, wrt="right") -> bool:
        """Returns true if the bounding box of the car is more than tolerance
        percentage outside of the road."""
        return self.oob_percentage(wrt=wrt) > tolerance

    def oob_distance(self,  wrt="right") -> float:
        """Returns the difference between the width of a lane and
        the distance between the car and the center of the road."""
        car_point = Point(self.car_pose)
        divisor = 4.0
        if wrt == "right":
            distance = self.road_polygon.right_polyline.distance(car_point)
        elif wrt == "left":
            distance = self.road_polygon.left_polyline.distance(car_point)
        else:
            distance = self.road_polygon.polyline.distance(car_point)
            divisor = 2.0
        difference = self.road_polygon.road_width / divisor - distance
        return difference

    def is_oob(self, wrt="right") -> bool:
        """Returns true if the car is an out-of-bound (OOB).

        The OOB can be calculated with respect to the left or right lanes,
        or with respect to the whole road.

        The car position is represented by the center of mass of the car.
        If you want to calculate the OOBs using the bounding box of the car,
        call self.is_oob_bb."""
        car_point = Point(self.car_pose)
        if wrt == "right":
            return not self.road_polygon.right_polygon.contains(car_point)
        elif wrt == "left":
            return not self.road_polygon.left_polygon.contains(car_point)
        else:
            return not self.road_polygon.polygon.contains(car_point)

    def _get_car_bbox_polygon(self) -> Polygon:
        car_bbox = self.car_bbox

        # x coordinates of the bounding box of the car.
        boundary_x = [
            car_bbox['near_bottom_left'][0],
            car_bbox['near_bottom_right'][0],
            car_bbox['far_bottom_right'][0],
            car_bbox['far_bottom_left'][0],
            car_bbox['near_bottom_left'][0],
        ]

        # y coordinates of the bounding box of the car.
        boundary_y = [
            car_bbox['near_bottom_left'][1],
            car_bbox['near_bottom_right'][1],
            car_bbox['far_bottom_right'][1],
            car_bbox['far_bottom_left'][1],
            car_bbox['near_bottom_left'][1],
        ]

        return Polygon(zip(boundary_x, boundary_y))
