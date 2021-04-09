import logging
from typing import List
from typing import Tuple

from shapely.geometry import Polygon, LineString

from self_driving.road_points import RoadPoints


class RoadPolygon:
    """A class that represents the road as a geometrical object
    (a polygon or a sequence of polygons)."""

    @classmethod
    def from_nodes(cls, nodes: List[Tuple[float, float, float, float]]):
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


if __name__ == '__main__':
    road_polygon = RoadPolygon.from_nodes([(0, 0, -28, 8),
                                           (0, 4, -28, 8),
                                           (5, 15, -28, 8),
                                           (20, -4, -28, 8)])

    assert not road_polygon.is_valid(), "It should be invalid"

    road_polygon = RoadPolygon.from_nodes([(0, 0, -28, 8),
                                           (3, 2, -28, 8),
                                           (10, -1, -28, 8)])

    assert road_polygon.is_valid(), "It should be valid"
