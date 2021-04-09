from typing import List, Tuple

import numpy as np

from self_driving.beamng_pose import BeamNGPose

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
        origin = np.add(p1[0:2], p1r[0:2]) / 2
        deg = np.degrees(np.arctan2([-v[0]], [-v[1]]))
        res = BeamNGPose(pos=tuple(origin + v) + (p1[2],), rot=(0, 0, deg[0]))
        return res

    def new_imagery(self):
        from .beamng_road_imagery import BeamNGRoadImagery
        return BeamNGRoadImagery(self)

    def plot_on_ax(self, ax):
        def _plot_xy(points, color, linewidth):
            tup = list(zip(*points))
            ax.plot(tup[0], tup[1], color=color, linewidth=linewidth)

        ax.set_facecolor('#7D9051')  # green
        _plot_xy(self.middle, '#FEA952', linewidth=1)  # arancio
        _plot_xy(self.left, 'white', linewidth=1)
        _plot_xy(self.right, 'white', linewidth=1)
        ax.axis('equal')


if __name__ == '__main__':

    road_edges_by_beamng = [
        {'right': [0, -4, -27.98419189453125], 'left': [0, 4, -27.98419189453125], 'middle': [0, 0, 0]},
        {'right': [20, -4, -27.98419189453125], 'left': [20, 4, -27.98419189453125], 'middle': [20, 0, 0]},
        {'right': [40, -4, -27.98419189453125], 'left': [40, 4, -27.98419189453125], 'middle': [40, 0, 0]},
        {'right': [62.828426361083984, -2.828427314758301, -27.98419189453125],
         'left': [57.171573638916016, 2.828427314758301, -27.98419189453125], 'middle': [60, 0, 0]},
        {'right': [84, 20, -27.98419189453125], 'left': [76, 20, -27.98419189453125], 'middle': [80, 20, 0]},
        {'right': [82.82843017578125, 42.828426361083984, -27.98419189453125],
         'left': [77.17156982421875, 37.171573638916016, -27.98419189453125], 'middle': [80, 40, 0]},
        {'right': [62.828426361083984, 62.828426361083984, -27.98419189453125],
         'left': [57.171573638916016, 57.171573638916016, -27.98419189453125], 'middle': [60, 60, 0]}]

    nodes = [(0, 0, -28, 8), (20, 0, -28, 8), (40, 0, -28, 8), (60, 0, -28, 8), (80, 20, -28, 8),
             (80, 40, -28, 8), (60, 60, -28, 8)]

    rd = RoadPoints.from_nodes(nodes)
    assert len(rd.middle) == len(road_edges_by_beamng)


    def distance(p1, p2):
        return np.linalg.norm(np.subtract((p1[0], p1[1]), (p2[0], p2[1])))


    max_dist = 0
    for i in range(len(rd.middle)):
        bng = road_edges_by_beamng[i]
        l = rd.left[i]
        r = rd.right[i]
        m = rd.middle[i]
        left_dist = distance(l, bng['left'])
        right_dist = distance(r, bng['right'])
        max_dist = max(left_dist, right_dist, max_dist)
        print('middle', bng['middle'])
        print('      ', m)
        print('   ', 'left  bng', bng['left'])
        print('   ', 'left calc', l)
        print('   ', 'right bng ', bng['right'])
        print('   ', 'right calc', r)

    print('max_dist', max_dist)
    assert max_dist < 0.0001
    print('success')
