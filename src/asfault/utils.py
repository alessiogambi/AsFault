import logging as l
import random

from asfault.tests import RoadTest,  get_start_goal_coords, get_path_polyline
from asfault.network import *

# TEST_ID = 0
# def next_test_id():
#     global TEST_ID
#     TEST_ID += 1
#     return TEST_ID
#
#
# def test_from_network(network):
#     start, goal, path = determine_start_goal_path(network)
#     l.debug('Got start, goal, and path for network.')
#     test = RoadTest(next_test_id(), network, start, goal)
#     if path:
#         l.debug('Setting path of new test: %s', test.test_id)
#         test.set_path(path)
#         l.debug('Set path of offspring.')
#     return test
#
#
# def determine_start_goal_path(network):
#     best_start, best_goal = None, None
#     best_path = None
#     best_score = -1
#
#     epsilon = 0.1
#     candidates = list(network.get_start_goal_candidates())
#     random.shuffle(candidates)
#     candidate_idx = 0
#
#     # sg_file = 'sg_{:08}.png'.format(self.sg_idx)
#     # sg_file = os.path.join(c.rg.get_plots_path(), sg_file)
#     # sg_json = 'sg_{:08}.json'.format(self.sg_idx)
#     # sg_json = os.path.join(c.rg.get_plots_path(), sg_json)
#     # with open(sg_json, 'w') as out_file:
#     # out_file.write(json.dumps(NetworkLayout.to_dict(network), indent=4, sort_keys=True))
#     # self.sg_idx += 1
#
#     # plot_network(sg_file, network)
#     if candidates:
#         for start, goal in candidates:
#             # l.info(sg_file)
#             l.info('Checking candidate: (%s, %s), %s/%s', start, goal, candidate_idx, len(candidates))
#             candidate_idx += 1
#             paths = network.all_paths(start, goal)
#             # paths = network.all_shortest_paths(start, goal)
#             start_coord, goal_coord = get_start_goal_coords(network, start, goal)
#             i = 0
#             done = 0.05
#             for path in paths:
#                 l.info('Path has length: %s', len(path))
#                 try:
#                     polyline = get_path_polyline(network, start_coord, goal_coord, path)
#                 except:
#                     break
#
#                 # TODO Select the best among the available paths?
#                 l.info('Got polyline.')
#                 # score = self.estimator.score_path(path, polyline)
#                 # l.info('Got score estimation: %s', score)
#                 # if score > best_score:
#
#                 best_start = start
#                 best_goal = goal
#                 best_path = path
#                 # best_score = score
#
#                 # i += 1
#                 #
#                 # done = self.rng.random()
#                 # if done < epsilon:
#                 break
#                 #
#                 # epsilon *= 1.25
#             # if done < epsilon:
#             break
#
#         best_start, best_goal = get_start_goal_coords(network, best_start, best_goal)
#
#         return best_start, best_goal, best_path
#
#     return None, None, None
