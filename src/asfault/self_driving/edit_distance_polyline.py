from typing import List, Tuple
import numpy as np

AngleLength = Tuple[float, float]
ListOfAngleLength = List[AngleLength]

Point = Tuple[float, float]
ListOfPoints = List[Point]


def _calc_cost_discrete(u: AngleLength, v: AngleLength):
    delta_angle, delta_len = np.subtract(u, v)
    # print(delta_angle)
    delta_angle = np.abs((delta_angle + 180) % 360 - 180)
    # print(str(delta_angle))
    eps_angle = 0.3
    eps_len = 0.2
    if delta_angle < eps_angle and delta_len < eps_len:
        res = 0
    else:
        res = 2

    # res = 1 / 2 * (delta_angle / (1 + delta_angle) + delta_len / (1 + delta_len))
    return res


def _calc_cost_weighted(u: AngleLength, v: AngleLength):
    delta_angle, delta_len = np.abs(np.subtract(u, v))
    delta_angle = np.abs((delta_angle + 180) % 360 - 180)
    eps_angle = 0.3
    eps_len = 0.2
    if delta_angle < eps_angle and delta_len < eps_len:
        res = 0
    else:
        res = 1 / 2 * (delta_angle / (1 + delta_angle) + delta_len / (1 + delta_len))
    return res


#_calc_cost = _calc_cost_discrete
_calc_cost = _calc_cost_weighted


def _iterative_levenshtein_dist_angle(s: ListOfAngleLength, t: ListOfAngleLength):
    """
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    rows = len(s) + 1
    cols = len(t) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            cost = _calc_cost(s[row - 1], t[col - 1])
            dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                 dist[row][col - 1] + 1,  # insertion
                                 dist[row - 1][col - 1] + cost)  # substitution
    # for r in range(rows):
    #     print(dist[r])

    return dist[row][col]


def _calc_angle_distance(v0, v1):
    at_0 = np.arctan2(v0[1], v0[0])
    at_1 = np.arctan2(v1[1], v1[0])
    return at_1 - at_0


def _calc_dist_angle(points: ListOfPoints) -> ListOfAngleLength:
    assert len(points) >= 2, f'at least two points are needed'

    def vector(idx):
        return np.subtract(points[idx + 1], points[idx])

    n = len(points) - 1
    result: ListOfAngleLength = [None] * (n)
    b = vector(0)
    for i in range(n):
        a = b
        b = vector(i)
        angle = np.degrees(_calc_angle_distance(a, b))
        distance = np.linalg.norm(b)
        result[i] = (angle, distance)
    return result


def iterative_levenshtein(s: ListOfPoints, t: ListOfPoints):
    s_da = _calc_dist_angle(s)
    t_da = _calc_dist_angle(t)
    # for i in range(len(s_da)):
    # if np.abs(np.subtract(s_da[i][0], t_da[i][0])) > 10:
    # print(str(s_da[i])+" "+str(t_da[i]))
    return _iterative_levenshtein_dist_angle(s_da, t_da)