from asfault.generator import RoadGenerator

import logging as l

class RepairRoadGenerator(RoadGenerator):


    # TODO Define a couple of methods to shrink the network, like clip (great a network from first and second intersecting
    #  segments) and trim (first to last intersecting segments)
    def __init__(self, bounds, seed):
        super().__init__(bounds, seed)

    def trim(self):
        # As far as I can tell the last segment must completely cross the boundaries
        #   that's why the network is not valid?

        # If we have more than 2 intersecting we trim until we get only 2 and check if the network is valid
        if len(self.network.get_boundary_intersecting_nodes()) > 2:
            todo = [self.root]
            road_segments_to_test = []
            while todo:
                todo_node = todo.pop(0)
                road_segments_to_test.append(todo_node)
                todo.extend([*self.network.parentage.successors(todo_node)])

            # Process the nodes/segments from the last to the first !
            for node in reversed(road_segments_to_test):
                l.info('Recovery action: Shrink road network.')
                self.network.remove_node(node)

                # from repair_testsuitegenerator import RepairTestSuiteGenerator
                # RepairTestSuiteGenerator.plot(self.network, 'trim')

                # Remove nodes until only 2 segments intersect the border and the last segment is
                if len(self.network.get_boundary_intersecting_nodes()) == 2:
                    break

        # At this point we have exactly 2 boundary
        if self.network.has_connected_boundary_segments():
            # We are done
            return RoadGenerator.done
        else:
            # Add one last segment "completely" outside the map

            counter = 0
            while self.grow() != RoadGenerator.done and counter < 10 :
                counter += 1

        self.seal_boundaries()
        return RoadGenerator.done

    def grow(self):
        # Grows the network to ensure that the road touches 2+ points on the boundary and the last segment is completely
        #   out of the map.
        l.info('Recovery: Attempting to grow road network.')
        ext_points = self.network.find_dead_ends()
        l.info('Recovery: Found %s extension points.', len(ext_points))

        growth = {}
        for ext_point in ext_points:
            extensions = self.extend(ext_point)
            if extensions:
                growth[ext_point] = extensions
            else:
                if len(self.extension_stack) == 0:
                    l.info('Recovery: No expansions. Give up')
                    return RoadGenerator.done

                self.shrink()
                return RoadGenerator.shrank

        if growth:
            for ext_point, extensions in growth.items():
                for extension in extensions:
                    self.network.add_parentage(ext_point, extension)

            # Add a new segment
            self.extension_stack.append(growth)
            self.network.update_abs()

            # Evalutate the growth
            # This does not work since we add and remove the last piece...
            # Branch was not long enough - Undo and Repeat
            # if not self.network.check_branch_lengths():
            #     self.shrink()
            #     return RoadGenerator.shrank

            # Not moving in the right direction - Undo and Repeat
            # TODO Check where's the goal
            if not self.is_goal_oriented(growth):
                l.info('Recovery: Expansion is not heading towards goal. Undoing.')
                self.shrink()
                return RoadGenerator.shrank
            # This might be relaxed
            # if not self.is_closer_to_goal(growth):
            #     l.info('Recovery: Expansion did not approach goal. Undoing.')
            #     self.shrink()
            #     return RoadGenerator.shrank

            # There are other problems - Undo and Repeat
            if not self.network.is_consistent():
                l.info('Recovery: Intersections found. Undoing..')
                self.shrink()
                return RoadGenerator.shrank

            # Check if we can stop or keep growing
            if len(self.network.get_boundary_intersecting_nodes()) > 1:
                if self.network.has_connected_boundary_segments():
                    return RoadGenerator.done
                else:
                    l.info('Recovery: Last segment not long enough..')
                    self.shrink()
                    return RoadGenerator.shrank
            else:
                l.info('Recovery: No intersections found, sealing off boundaries.')
                self.seal_boundaries()
                return RoadGenerator.grown

        # We cannot grow somehow?
        return RoadGenerator.done
