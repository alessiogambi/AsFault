import logging as l

from asfault.network import *
from asfault.plotter import *

from time import time

import random

def plot_network(plot_file, network):
    title = 'debug'
    plotter = StandaloneTestPlotter(title, network.bounds)
    plotter.plot_network(network)
    save_plot(plot_file, dpi=c.pt.dpi_final)


class MetaCrossover:
    """ It combines the logic that retries the crossover operation if fails """
    # For the moment we use only Join
    def __init__(self):
        self.choice = Join()

    def attempt_crosssover(self, mom, dad, crossover):
        l.debug('Attempting to cross over: %s (%s) %s', mom, type(crossover), dad)
        epsilon = 0.01
        while True:
            children, aux = crossover.apply(mom, dad)
            if children:
                consistent = []
                for child in children:
                    try:
                        if child.complete_is_consistent():
                            consistent.append(child)
                        else:
                            break
                    except Exception as e:
                        l.error('Exception while creating test from child: ')
                        # l.exception(e)
                        break

                if consistent:
                    return consistent, aux

            failed = random.random()
            if failed < epsilon:
                break

            epsilon *= 1.05

        return None, aux

    def crossover(self, mom, dad):
        children, aux = self.choice.try_all(mom, dad, self)
        l.info('Finished trying all crossovers')

        # children, aux = self.attempt_crosssover(mom, dad, choice)
        if not aux:
            aux = {}
        if children:
            aux['type'] = self.choice
            l.debug('Cross over was applicable.')
            tests = []
            for child in children:
                if child.complete_is_consistent():
                    test = self.test_from_network(child)
                    tests.append(test)
            return tests, aux

        l.debug('Cross over between %s x %s considered impossible.', mom, dad)
        return None, {}


class Crossover:
    def __init__(self, name):
        self.name = name

    def apply(self, mom, dad):
        raise NotImplementedError()

    def try_all(self, mom, dad, gen):
        raise NotImplementedError()


class Merge(Crossover):
    NAME = 'merge'

    def __init__(self, name=None):
        if not name:
            name = Merge.NAME
        super().__init__(name)

    def get_point_side(self, line, point):
        side = (point[0] - line.coords[0][0]) * \
               (line.coords[-1][1] - line.coords[0][1]) - \
               (point[1] - line.coords[0][1]) * \
               (line.coords[-1][0] - line.coords[0][0])

        if side < 0:
            return -1
        if side > 0:
            return 1
        return 0

    def is_applicable(self, mom, dad):
        intersected = False
        for m_node in mom.parentage.nodes():
            if m_node.roadtype in GHOST_TYPES:
                continue

            m_poly = m_node.abs_polygon
            intersecting = dad.get_intersecting_nodes(m_poly)
            if len(intersecting) > 1:
                return False
            if intersecting:
                other = set(intersecting)
                other = other.pop()
                m_spine = m_node.get_spine()
                d_spine = other.get_spine()
                intersection = m_spine.intersection(d_spine)
                if intersection.geom_type != 'Point':
                    return False

            if intersecting:
                intersected = True

        return intersected

    def apply_networks(self, mom, dad):
        if not self.is_applicable(mom, dad):
            return None, {}

        m_copy = mom.copy()
        m_copy.merge(dad)
        return [m_copy], {}

    def apply(self, mom, dad):
        mom = mom.network
        dad = dad.network
        return self.apply_networks(mom, dad)


class PartialMerge(Merge):
    NAME = 'partial_merge_{}_{}'

    def __init__(self, rng, m_count='rng', d_count='rng'):
        super().__init__(rng, PartialMerge.NAME.format(m_count, d_count))

        self.m_count = m_count
        self.d_count = d_count

    def trim_network(self, network, keep):
        trimmed = network.copy()
        roots = trimmed.get_roots()
        trim = {root for root in roots if root not in keep}
        for root in trim:
            trimmed.remove_after(root)
            trimmed.remove_node(root)
        return trimmed

    def split_network(self, network, count):
        roots = network.get_roots()

        take = random.sample(roots, count)
        leave = {node for node in roots if node not in take}

        taken = self.trim_network(network, take)
        left = None
        if leave:
            left = self.trim_network(network, leave)

        return taken, left

    def apply_counts(self, m_network, d_network, m_count, d_count):
        m_taken, m_left = self.split_network(m_network, m_count)
        d_taken, d_left = self.split_network(d_network, d_count)

        assert m_taken
        assert d_taken

        ret, aux = self.apply_networks(m_taken, d_taken)
        if not ret:
            return None, {}

        d_offspring = None
        if m_left:
            d_offspring = m_left
        if d_left:
            d_offspring = d_left
        if m_left and d_left:
            d_offspring, aux = self.apply_networks(m_left, d_left)
            if d_offspring:
                d_offspring = d_offspring[0]

        if d_offspring:
            ret.append(d_offspring)

        return ret, {}

    def apply(self, mom, dad):
        m_network = mom.network
        d_network = dad.network
        m_roots = m_network.get_roots()
        d_roots = d_network.get_roots()

        if self.m_count == 'rng':
            m_count = random.randint(1, len(m_roots))
        else:
            m_count = self.m_count
        if self.d_count == 'rng':
            d_count = random.randint(1, len(d_roots))
        else:
            d_count = self.d_count

        if m_count > len(m_roots):
            return None, {}
        if d_count > len(d_roots):
            return None, {}

        return self.apply_counts(m_network, d_network, m_count, d_count)


    def try_all(self, mom, dad, gen):
        m_network = mom.network
        d_network = dad.network
        m_roots = m_network.get_roots()
        d_roots = d_network.get_roots()

        m_range = list(range(1, len(m_roots) + 1))
        d_range = list(range(1, len(d_roots) + 1))
        random.shuffle(m_range)
        random.shuffle(d_range)

        for m_count in m_range:
            for d_count in d_range:
                results, aux = self.apply_counts(m_network, d_network, m_count, d_count)
                if results:
                    certified = []
                    for result in results:
                        if result.complete_is_consistent():
                            try:
                                # Do generate the test template inside crossover. Do it only later.
                                # This assumes that no additional checks are performed during generate_test_prefab or gen.test_from_network
                                # test = gen.test_from_network(result)
                                # generate_test_prefab(test)
                                certified.append(result)
                            except Exception as e:
                                l.error('Exception while crossing over: ')
                                #l.exception(e)
                                break

                    if len(certified) == len(results):
                        return certified, aux

        return None, {}


class Join(Crossover):
    def __init__(self):
        super().__init__('join')

    def perform_join(self, mom, dad, m_joint, d_joint):
        l.debug('Selected mom joint %s and dad joint %s', m_joint, d_joint)
        m_cut, m_joint = mom.cut_branch(m_joint)
        m_cut.join(m_joint, dad, d_joint)
        l.debug('Joined networks at the selected joints.')

        return m_cut, {'m_joint': m_joint, 'd_joint': d_joint}

    def join(self, mom, dad):
        mom = mom.network
        dad = dad.network

        m_nodes = {node for node in mom.parentage.nodes() if
                   node.roadtype not in GHOST_TYPES}
        d_nodes = {node for node in dad.parentage.nodes() if
                   node.roadtype not in GHOST_TYPES}

        l.debug('Got mom and dad nodes for joint selection.')

        m_joint = random.sample(m_nodes, 1)[0]
        d_joint = random.sample(d_nodes, 1)[0]

        return self.perform_join(mom, dad, m_joint, d_joint)

    def apply(self, mom, dad):
        akid, aaux = self.join(mom, dad)
        bkid, baux = self.join(dad, mom)
        ret = []
        aux = {}
        if akid and akid.complete_is_consistent():
            ret.append(akid)
            aux['aaux'] = aaux
        if bkid and bkid.complete_is_consistent():
            ret.append(bkid)
            aux['baux'] = baux

        if len(ret) != 2:
            return None, {}

        return ret, aux

    def certify(self, network, gen):
        if network.complete_is_consistent():
            try:
                # Is this really necessary
                gen.test_from_network(network)
                return True
            except Exception as e:
                l.error('Exception while joining: ')
                #l.exception(e)
        return False

    def try_all(self, mom, dad, gen):
        mom = mom.network
        dad = dad.network

        m_roots = list(mom.get_roots())
        d_roots = list(dad.get_roots())

        random.shuffle(m_roots)
        random.shuffle(d_roots)

        attempts = 0
        for m_root in m_roots:
            m_branch = mom.get_branch_from(m_root)
            m_branch = m_branch[1:]
            m_beg = math.floor(len(m_branch) * 0.25)
            m_end = math.ceil(len(m_branch) * 0.75)
            assert m_beg < m_end
            m_branch = m_branch[m_beg:m_end]
            random.shuffle(m_branch)

            for d_root in d_roots:
                d_branch = dad.get_branch_from(d_root)
                d_branch = d_branch[1:]
                d_beg = math.floor(len(d_branch) * 0.25)
                d_end = math.ceil(len(d_branch) * 0.75)
                assert d_beg < d_end
                d_branch = d_branch[d_beg:d_end]
                random.shuffle(d_branch)

                m_eps = 0.5
                for m_joint in m_branch:
                    d_eps = 0.5
                    for d_joint in d_branch:
                        attempts += 1
                        l.info('Join attempt # %s', attempts)
                        certified = []
                        aux = {}
                        m_copy = mom.copy()
                        d_copy = dad.copy()
                        m_child, a_aux = self.perform_join(m_copy, d_copy, m_joint, d_joint)
                        if self.certify(m_child, gen):
                            certified.append(m_child)
                            aux['aaux'] = a_aux
                        m_copy = mom.copy()
                        d_copy = dad.copy()
                        d_child, b_aux = self.perform_join(d_copy, m_copy, d_joint, m_joint)
                        if self.certify(d_child, gen):
                            certified.append(d_child)
                            aux['baux'] = b_aux

                        if len(certified) == 2:
                            return certified, aux

                        failed = random.random()
                        if True or failed  < d_eps:
                            break
                        d_eps *= 1.25

                    failed = random.random()
                    if True or failed < m_eps:
                        break
                    m_eps *= 1.25

        return None, {}
