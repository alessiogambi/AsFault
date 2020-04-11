# import math
#
# from asfault.crossovers import Join
# from asfault.generator import Point
# from numpy.random.mtrand import randint
# from shapely.geometry import box, LineString
#
# from asfault.repair_roadgenerator import RepairRoadGenerator
# from asfault.repair_testsuitegenerator import RepairTestSuiteGenerator
#
# import shapely.ops as so
# from shapely.affinity import translate, rotate
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
# import logging as l
#
# import sys
#
# # Define how much the road should be moved outside the map to ensure the cars will be completely on the generated roads
# BOUNDARY_OVERLAP = 20
#
# # TODO: https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
# class RepairJoin(Join):
#
#     def __init__(self, rng, env_size):
#         Join.__init__(self, rng)
#         self.name = "repair-join"
#         self.env_size = env_size
#
#     def try_all(self, mom_test, dad_test, gen):
#         mom_network = mom_test.network
#         dad_network = dad_test.network
#
#         m_roots = list(mom_network.get_roots())
#         d_roots = list(dad_network.get_roots())
#
#         self.rng.shuffle(m_roots)
#         self.rng.shuffle(d_roots)
#
#         attempts = 0
#         for m_root in m_roots:
#             m_branch = mom_network.get_branch_from(m_root)
#             m_branch = m_branch[1:]
#             m_beg = math.floor(len(m_branch) * 0.25)
#             m_end = math.ceil(len(m_branch) * 0.75)
#             assert m_beg < m_end
#             m_branch = m_branch[m_beg:m_end]
#             self.rng.shuffle(m_branch)
#
#             for d_root in d_roots:
#                 d_branch = dad_network.get_branch_from(d_root)
#                 d_branch = d_branch[1:]
#                 d_beg = math.floor(len(d_branch) * 0.25)
#                 d_end = math.ceil(len(d_branch) * 0.75)
#                 assert d_beg < d_end
#                 d_branch = d_branch[d_beg:d_end]
#                 self.rng.shuffle(d_branch)
#
#                 m_eps = 0.5
#                 for m_joint in m_branch:
#                     d_eps = 0.5
#
#                     for d_joint in d_branch:
#                         attempts += 1
#
#                         l.info('REPAIR: Join attempt # %s', attempts)
#
#                         certified_networks = []
#                         aux = {}
#                         copy_of_mom_network = mom_network.copy()
#                         copy_of_dad_network = dad_network.copy()
#
#                         # This does the actual join of the networks
#                         m_child, a_aux = self.perform_join(copy_of_mom_network, copy_of_dad_network, m_joint, d_joint)
#
#                         if self.certify(m_child, gen):
#                             certified_networks.append(m_child)
#                             aux['aaux'] = a_aux
#                         else:
#                             # At this point the network is broken we attempt a repair
#                             if RepairTestSuiteGenerator.onlyNotTouchingBoundaries(m_child):
#
#                                 # The repair might have fixed the problem with the network, but the network might have
#                                 #   become inconsisten in other aspects so we need to double check that
#                                 m_child = self.repair(m_child)
#
#                                 if m_child is not None:
#                                     if self.certify(m_child, gen):
#                                         certified_networks.append(m_child)
#                                         aux['aaux'] = a_aux
#                                     else:
#                                         l.warning("REPAIR: Join repair was not effective (m_child)")
#                             else:
#                                 l.warning("REPAIR: Join repair cannot be applied")
#
#                         copy_of_mom_network = mom_network.copy()
#                         copy_of_dad_network = dad_network.copy()
#
#                         d_child, b_aux = self.perform_join(copy_of_dad_network, copy_of_mom_network, d_joint, m_joint)
#
#                         if self.certify(d_child, gen):
#                             certified_networks.append(d_child)
#                             aux['baux'] = b_aux
#                         else:
#                             if RepairTestSuiteGenerator.onlyNotTouchingBoundaries(d_child):
#
#                                 d_child = self.repair(d_child)
#
#                                 if d_child is not None:
#                                     if self.certify(d_child, gen):
#                                         certified_networks.append(d_child)
#                                         aux['baux'] = b_aux
#                                     else:
#                                         l.warning("REPAIR: Join repair was not effective (d_child)")
#                             else:
#                                 l.warning("REPAIR: Join repair cannot be applied")
#
#                         if len(certified_networks) == 2:
#                             return certified_networks, aux
#
#                         # Decide whether we want to retry the cross-over again or give up
#                         failed = self.rng.random()
#                         if True or failed < d_eps:
#                             break
#                         d_eps *= 1.25
#
#                         # Decide whether we want to retry the cross-over again or give up
#                     failed = self.rng.random()
#                     if True or failed < m_eps:
#                         break
#                     m_eps *= 1.25
#
#         # Not even attempted a cross over?
#         return None, {}
#
#
#     def plot_network_segments_and_bounding_box(self, network):
#         ax = plt.gca()
#
#         # Plotting MAP BOUNDARY
#         the_map_bounding_box= network.bounds
#         x, y = the_map_bounding_box.exterior.xy
#         ax.plot(x, y, color='green')
#
#         # Set X/Y lims around the MAP BOUNDARY
#         x_lim, y_lim = the_map_bounding_box.buffer(100).exterior.xy
#         ax.set_xlim(min(x_lim), max(x_lim))
#         ax.set_ylim(min(y_lim), max(y_lim))
#
#         # Plotting road segments.
#         # network.nodes is a dictionary !
#         for node in network.nodes.values():
#             if node.abs_polygon:
#                 plt.plot(*node.abs_polygon.exterior.xy)
#
#         # Compute the union of the polygons defining the road
#         new_shape = so.cascaded_union([node.abs_polygon for node in network.nodes.values() if node.abs_polygon])
#
#         # Compute the bounds over the road and plot them
#         rect = patches.Rectangle((new_shape.bounds[0], new_shape.bounds[1]),
#                                  (new_shape.bounds[2] - new_shape.bounds[0]),
#                                  (new_shape.bounds[3] - new_shape.bounds[1]), linewidth=1, edgecolor='r',
#                                  facecolor='none')
#         ax.add_patch(rect)
#
#         # Plot the "exact" starting point of this road
#         the_road_root = list(network.get_roots())[0]
#         starting_point = Point(the_road_root.x_off, the_road_root.y_off)
#         ax.plot(starting_point.x, starting_point.y, marker="o")
#
#     def compute_overlap_area_at(self, target_point, network):
#         # Compute the union of the polygons defining the road and the bounding box around the entire road
#         # TODO This is always the same maybe we can reuse it?
#
#         the_road = so.cascaded_union([node.abs_polygon for node in network.nodes.values() if node.abs_polygon])
#         the_road_root = list(network.get_roots())[0]
#         starting_point = Point(the_road_root.x_off, the_road_root.y_off)
#
#         # Despite its name network.bounds is a POLYGON
#         the_map_bounding_box = network.bounds
#
#         # Compute the translation vector from network starting point to the given target_point
#         translate_by = np.array(target_point.coords) - np.array(starting_point.coords)
#
#         # Translate the poligon of the road to the target point
#         translated_road = translate(the_road, xoff=translate_by[0][0], yoff=translate_by[0][1])
#
#         translated_road_bounding_box = box(translated_road.bounds[0], translated_road.bounds[1], translated_road.bounds[2],
#             translated_road.bounds[3])
#
#
#         # TODO We approximate the area of the road with the area of its bounding box since we shapely does not let us easily
#         #   compute the area (it raises an exception?)
#         return the_map_bounding_box.intersection(translated_road_bounding_box).area
#
#     def repair(self, network):
#         try:
#             l.warning('REPAIR: Join attempt a Repair')
#
#             # Enable plotting with a flag
#             # RepairTestSuiteGenerator.plot(network, 'broken')
#
#             # We move the broken road around such that its starting point is on the border and we get largest map coverage
#             # (note we use the bbox instead of the actual road polygons)
#             # import matplotlib.pyplot as plt
#             # plt.figure('x-ray')
#             # plt.cla()
#             # self.plot_network_segments_and_bounding_box(network)
#
#             # First rank the edges and then explore the best one
#             x, y = network.bounds.exterior.coords[0]
#             x = math.fabs(x)
#             y = math.fabs(y)
#
#             edges = [
#                 # N
#                 [(-x, +y), (+x, +y)],
#                 # E
#                 [(+x, -y), (+x, +y)],
#                 # S
#                 [(-x, -y), (+x, -y)],
#                 # W
#                 [(-x, -y), (-x, +y)]
#             ]
#
#             # In some occasions compute_overlap_area_at triggers an exception. We capture it and fail the repair
#             #   attempt instead of breaking AsFault
#             max_area = -1
#             best_edge = None
#             for edge in edges:
#                 edge_mid_point = Point((edge[0][0] + edge[1][0]) / 2.0, (edge[0][1] + edge[1][1]) / 2.0)
#                 area = self.compute_overlap_area_at(edge_mid_point, network)
#                 if area > max_area:
#                     max_area = area
#                     best_edge = edge
#
#             # At this point we do a simple linear search over the edge to identify where we should place the road
#             # starting point to maximize the area
#
#             if best_edge[0][0] == best_edge[1][0]:
#                 # if X are the same, then we lin space over Y
#                 target_points = [Point(best_edge[0][0], y) for y in np.linspace(best_edge[0][1], best_edge[1][1], 10)]
#             else:
#                 # if Y are the same, then we lin space over X
#                 target_points = [Point(x, best_edge[0][1]) for x in np.linspace(best_edge[0][0], best_edge[1][0], 10)]
#
#             best_location = None
#             max_area = -1
#             for target_point in target_points:
#                 area = self.compute_overlap_area_at(target_point, network)
#                 if area > max_area:
#                     max_area = area
#                     best_location = target_point
#
#             # At this point the road starts from a boundary, we need to check whether the road must grow or shrink
#             # First we move the road to the target point ensuring that the root is slightly outside the map so cars will
#             # be completely inside the road when tests start
#
#             # Assumption: Road has ONLY one street so there's only one root
#             root_node = list(network.get_roots())[0]
#
#             # 0 deg is (0,1) not (1,0) !
#             seg = LineString([(0, 0), (0, BOUNDARY_OVERLAP)])
#             # Rotate it by the angle (degree) defined by root but in the opposite direction +180
#             seg = rotate(seg, root_node.angle+180, origin=Point(0,0), use_radians=False)
#             # Translate the segment in the best location
#             seg = translate(seg, xoff=best_location.x, yoff=best_location.y)
#             # Now move the root node into the new location
#             root_node.x_off = seg.coords[1][0]
#             root_node.y_off = seg.coords[1][1]
#
#             # Recompute the polygons of the entire road
#             network.update_abs(force=True)
#
#             # TODO Enable with a flag
#             # RepairTestSuiteGenerator.plot(network, 'half-repaired')
#
#             # Check how many segments in the road touch the boundary
#             boundary_crossing_segments = len(network.get_boundary_intersecting_nodes())
#
#             # This should never happen since we have just moved the road on the border...
#             assert boundary_crossing_segments > 0, "Something wrong happened, the translated road does not touch " \
#                                                         "any border ?!"
#
#             if boundary_crossing_segments == 1:
#                 # We need to use a generator to grow and shrink the road as it has all the methods already available
#                 repair_generator = RepairRoadGenerator(network.bounds, randint(10000))
#                 repair_generator.network = network
#                 # Assume we have only one road
#                 bestNode = list(network.get_roots())[0]
#                 repair_generator.root = bestNode
#
#                 counter = 0
#                 while repair_generator.grow() != repair_generator.done and counter < 10 :
#                     counter += 1
#
#                 # Is this really necessay?!
#                 network = repair_generator.network
#
#             else:
#                 repair_generator = RepairRoadGenerator(network.bounds, randint(10000))
#                 repair_generator.network = network
#                 bestNode = list(network.get_roots())[0]
#                 repair_generator.root = bestNode
#
#                 repair_generator.trim()
#
#                 # Is this really necessay?!
#                 network = repair_generator.network
#
#             # TODO Enable with a flag
#             # This might be broken
#             # RepairTestSuiteGenerator.plot(network, 'repaired')
#             #
#             # plt.figure('repaired-x-ray')
#             # plt.cla()
#             # self.plot_network_segments_and_bounding_box(network)
#
#             # Check that all the other properties of valid networks hold
#             network.update_abs(force=True)
#             # Not sure if force makes a difference
#             if network.complete_is_consistent():
#                 l.warning("REPAIR: Join Repair worked")
#                 return network
#             else:
#                 l.warning("REPAIR: Join Repair failed")
#                 return None
#         except:
#             # Catch-all clause
#             l.warning("REPAIR: Join Repair failed with exception", sys.exc_info()[0])
#             return None
#
