# from asfault.evolver import TestSuiteGenerator
#
# class RepairTestSuiteGenerator(TestSuiteGenerator):
#
#     # def __init__(self, env_size):
#     #     TestSuiteGenerator.__init__(self)
#     #     self.joiner = repair_crossover.RepairJoin(self.rng)
#     #     self.env_size = env_size
#
#     # def crossover(self, mom, dad):
#     #     choice = self.rng.random()
#     #     choices = []
#     #
#     #     # l.debug('Picking crossover %s < %s ?', choice, c.ev.join_probability)
#     #     if choice < c.ev.join_probability:
#     #         choices = [self.joiner]
#     #         l.info('Crossover using for %s', choice)
#     #     else:
#     #         if c.ev.try_all_ops:
#     #             choices.append(self.merger)
#     #             l.info('Crossover using for %s', choice)
#     #         else:
#     #             l.info('Skip Crossover')
#     #
#     #     for choice in choices:
#     #         children, aux = choice.try_all(mom, dad, self)
#     #         l.info('Finished trying all crossovers')
#     #
#     #         #children, aux = self.attempt_crosssover(mom, dad, choice)
#     #         if not aux:
#     #             aux = {}
#     #         if children:
#     #             aux['type'] = choice
#     #             l.debug('Cross over was applicable.')
#     #             tests = []
#     #             for child in children:
#     #                 if child.complete_is_consistent():
#     #                     test = self.test_from_network(child)
#     #                     tests.append(test)
#     #                 # ATTEMPT THE REPAIR
#     #                 elif self.onlyNotTouchingBoundaries(child):
#     #                     l.warning("Attempting repair of broken child %s", child)
#     #                     repairedChild = self.repair(child)
#     #                     if repairedChild != None:
#     #                         test = self.test_from_network(repairedChild)
#     #                         tests.append(test)
#     #             return tests, aux
#     #
#     #     l.debug('Cross over between %s x %s considered impossible.', mom, dad)
#     #     return None, {}
#     #
#     # def repair(self, network):
#     #     #self.plot(network)
#     #     # TODO: This was tested only for one road. We must generalize to maps and select only the broken road
#     #     bestNode = network.get_roots()[0]
#     #
#     #     generator = RepairRoadGenerator(box(-self.env_size, -self.env_size, self.env_size, self.env_size), randint(10000))
#     #     generator.network = network
#     #     generator.root = bestNode
#     #     while generator.grow() != generator.done:
#     #         pass
#     #
#     #     if generator.network.complete_is_consistent():
#     #         return generator.network
#     #     elif self.onlyNotTouchingBoundaries(generator.network):
#     #         return self.repair(generator.network)
#     #     else:
#     #         return None
#     #
#     @staticmethod
#     def onlyNotTouchingBoundaries(network):
#         if not network.is_consistent():
#             return False
#
#         if not network.clean_intersection_check():
#             return False
#
#         if not network.all_branches_connected():
#             return False
#
#         if not network.check_branch_lengths():
#             return False
#
#         if not network.has_connected_boundary_segments():
#             return True
#
#         return False
#
#     # @staticmethod
#     # def plot(network, fig_id):
#     #     # Do not plot for the moment TODO Control this with flag
#     #     # l.info("Plot network")
#     #     # Access the last figure and clear it out
#     #     # fig = plt.figure(fig_id, clear=True)
#     #     # ax = fig.subplots(nrows=1, ncols=1, sharey=True, squeeze=True)
#     #     # asfault_plotter = TestPlotter(ax, "Test", network.bounds )
#     #     # asfault_plotter.plot_network(network)
#     #     # plt.show()
#     #     l.debug("Plotting is disabled for repair")
#     #     pass
#     #
#     # @staticmethod
#     # def plotPoint(network, point):
#     #     # Do not plot for the moment TODO Control this with flag
#     #     # l.info("plotPoint")
#     #     # pass
#     #     # fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, squeeze=True)
#     #     # asfault_plotter = TestPlotter(ax, "{0:.4f}",
#     #     #                               network.bounds)
#     #     # asfault_plotter.plot_network(network)
#     #     # asfault_plotter.place_goal(point)
#     #     # plt.show()
#     #     l.debug("PlotPoint is disabled for repair")
