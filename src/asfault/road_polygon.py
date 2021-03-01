import logging
from typing import List
from typing import Tuple

from shapely.geometry import Polygon, LineString

from typing import List, Tuple

import numpy as np

class BeamNGPose:
    def __init__(self, pos=None, rot=None):
        self.pos = pos if pos else (0, 0, 0)
        self.rot = rot if rot else (0, 0, 0)

List4DTuple = List[Tuple[float, float, float, float]]
List2DTuple = List[Tuple[float, float]]


class RoadPoints:

    @classmethod
    def from_nodes(cls, middle_nodes: List4DTuple):
        res = RoadPoints()
        res.add_middle_nodes(middle_nodes)
        return res

    def __init__(self):
        self.middle: List4DTuple = []
        self.right: List2DTuple = []
        self.left: List2DTuple = []
        self.n = 0

    def add_middle_nodes(self, middle_nodes):
        n = len(self.middle) + len(middle_nodes)

        assert n >= 2, f'At least, two nodes are needed'

        assert all(len(point) >= 4 for point in middle_nodes), \
            f'A node is a tuple of 4 elements (x,y,z,road_width)'

        self.n = n
        self.middle += list(middle_nodes)
        self.left += [None] * len(middle_nodes)
        self.right += [None] * len(middle_nodes)
        self._recalculate_nodes()
        return self

    def _recalculate_nodes(self):
        for i in range(self.n - 1):
            l, r = self.calc_point_edges(self.middle[i], self.middle[i + 1])
            self.left[i] = l
            self.right[i] = r

        # the last middle point
        self.right[-1], self.left[-1] = self.calc_point_edges(self.middle[-1], self.middle[-2])

    @classmethod
    def calc_point_edges(cls, p1, p2) -> Tuple[Tuple, Tuple]:
        origin = np.array(p1[0:2])

        a = np.subtract(p2[0:2], origin)

        #TODO: changed from 2 to 4
        # calculate the vector which length is half the road width
        v = (a / np.linalg.norm(a)) * p1[3] / 2
        # add normal vectors
        l = origin + np.array([-v[1], v[0]])
        r = origin + np.array([v[1], -v[0]])
        return tuple(l), tuple(r)

    def vehicle_start_pose(self, meters_from_road_start=2.5, road_point_index=0) \
            -> BeamNGPose:
        assert self.n > road_point_index, f'road length is {self.n} it does not have index {road_point_index}'
        p1 = self.middle[road_point_index]
        p1r = self.right[road_point_index]
        p2 = self.middle[road_point_index + 1]

        p2v = np.subtract(p2[0:2], p1[0:2])
        v = (p2v / np.linalg.norm(p2v)) * meters_from_road_start
        # TODO: test the amount
        origin = np.add(p1[0:2], p1r[0:2]) / 2
        deg = np.degrees(np.arctan2([-v[0]], [-v[1]]))
        res = BeamNGPose(pos=tuple(origin + v) + (p1[2],), rot=(0, 0, deg[0]))
        return res

    def plot_on_ax(self, ax):
        def _plot_xy(points, color, linewidth):
            tup = list(zip(*points))
            ax.plot(tup[0], tup[1], color=color, linewidth=linewidth)

        ax.set_facecolor('#7D9051')  # green
        _plot_xy(self.middle, '#FEA952', linewidth=1)  # arancio
        _plot_xy(self.left, 'white', linewidth=1)
        _plot_xy(self.right, 'white', linewidth=1)
        ax.axis('equal')


class RoadPolygon:
    """A class that represents the road as a geometrical object
    (a polygon or a sequence of polygons)."""

    @classmethod
    def from_nodes(cls, nodes: List[Tuple[float, float]]):
        return RoadPolygon(RoadPoints.from_nodes(nodes))

    def __init__(self, road_points: RoadPoints):
        assert len(road_points.left) == len(road_points.right) == len(road_points.middle)
        assert len(road_points.left) >= 2
        assert all(len(x) == 4 for x in road_points.middle)
        assert all(len(x) == 2 for x in road_points.left)
        assert all(len(x) == 2 for x in road_points.right)
        # assert all(x[3] == road_points.middle[0][3] for x in
        #            road_points.middle), "The width of the road should be equal everywhere."
        self.road_points = road_points
        self.road_width = road_points.middle[0][3]
        self.polygons = self._compute_polygons()
        self.polygon = self._compute_polygon()
        self.right_polygon = self._compute_right_polygon()
        self.left_polygon = self._compute_left_polygon()
        self.polyline = self._compute_polyline()
        self.right_polyline = self._compute_right_polyline()
        self.left_polyline = self._compute_left_polyline()
        self.num_polygons = len(self.polygons)

    def _compute_polygons(self) -> List[Polygon]:
        """Creates and returns a list of Polygon objects that represent the road.
        Each polygon represents a segment of the road. Two objects adjacent in
        the returned list represent adjacent segments of the road."""
        polygons = []
        for left, right, left1, right1, in zip(self.road_points.left,
                                               self.road_points.right,
                                               self.road_points.left[1:],
                                               self.road_points.right[1:]):
            assert len(left) >= 2 and len(right) >= 2 and len(left1) >= 2 and len(right1) >= 2
            # Ignore the z coordinate.
            polygons.append(Polygon([left[:2], left1[:2], right1[:2], right[:2]]))
        return polygons

    def _compute_polygon(self) -> Polygon:
        """Returns a single polygon that represents the whole road."""
        road_poly = self.road_points.left.copy()
        road_poly.extend(self.road_points.right[::-1])
        return Polygon(road_poly)

    def _compute_right_polygon(self) -> Polygon:
        """Returns a single polygon that represents the right lane of the road."""
        road_poly = [(p[0], p[1]) for p in self.road_points.middle]
        road_poly.extend(self.road_points.right[::-1])
        return Polygon(road_poly)

    def _compute_left_polygon(self) -> Polygon:
        """Returns a single polygon that represents the left lane of the road."""
        road_poly = self.road_points.left.copy()
        road_poly.extend([(p[0], p[1]) for p in self.road_points.middle][::-1])
        return Polygon(road_poly)

    def _compute_polyline(self) -> LineString:
        """Computes and returns a LineString representing the polyline
        of the spin (or middle) of the road."""
        return LineString([(n[0], n[1]) for n in self.road_points.middle])

    def _compute_right_polyline(self) -> LineString:
        """Computes and returns a LineString representing the polyline
        of the spin (or middle) of the right lane of the road."""
        return LineString([((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2) for p1, p2 in
                           zip(self.road_points.middle, self.road_points.right)])

    def _compute_left_polyline(self) -> LineString:
        """Computes and returns a LineString representing the polyline
        of the spin (or middle) of the left lane of the road."""
        return LineString([((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2) for p1, p2 in
                           zip(self.road_points.left, self.road_points.middle)])

    def _get_neighbouring_polygons(self, i: int) -> List[int]:
        """Returns the indices of the neighbouring polygons of the polygon
        with index i."""
        if self.num_polygons == 1:
            assert i == 0
            return None
        assert 0 <= i < self.num_polygons
        if i == 0:
            return [i + 1]
        elif i == self.num_polygons - 1:
            return [i - 1]
        else:
            assert self.num_polygons >= 3
            return [i - 1, i + 1]

    def _are_neighbouring_polygons(self, i: int, j: int) -> bool:
        """Returns true if the polygons represented by the indices i and j are adjacent."""
        return j in self._get_neighbouring_polygons(i)

    def is_valid(self) -> bool:
        """Returns true if the current RoadPolygon representation of the road is valid,
        that is, if there are no intersections between non-adjacent polygons and if
        the adjacent polygons have as intersection a LineString (a line or segment)."""
        if self.num_polygons == 0:
            logging.debug("No polygon constructed.")
            return False

        for i, polygon in enumerate(self.polygons):
            if not polygon.is_valid:
                logging.debug("Polygon %s is invalid." % polygon)
                return False

        for i, polygon in enumerate(self.polygons):
            for j, other in enumerate(self.polygons):
                # Ignore the case when other is equal to the polygon.
                if other == polygon:
                    assert i == j
                    continue
                if polygon.contains(other) or other.contains(polygon):
                    logging.debug("No polygon should contain any other polygon.")
                    return False
                if not self._are_neighbouring_polygons(i, j) and other.intersects(polygon):
                    logging.debug("The non-neighbouring polygons %s and %s intersect." % (polygon, other))
                    return False
                if self._are_neighbouring_polygons(i, j) and not isinstance(other.intersection(polygon), LineString):
                    logging.debug("The neighbouring polygons %s and %s have an intersection of type %s." % (
                        polygon, other, type(other.intersection(polygon))))
                    return False
        logging.debug("The road is apparently valid.")
        return True

