import logging as l
from asfault.network import *


class Mutator:
    def __init__(self, name, rng):
        self.name = name
        self.rng = rng

    def apply(self, resident):
        raise NotImplementedError()


class SegmentReplacingMutator(Mutator):
    def __init__(self, rng, name, key, factory):
        super().__init__(name, rng)
        self.key = key
        self.factory = factory

    def get_target(self, network):
        options = [option for option in network.nodes.values(
        ) if option.roadtype not in GHOST_TYPES]
        options = [option for option in options if option.key != self.key]
        if options:
            return self.rng.choice(options)
        return None

    def apply(self, resident):
        network = resident.network.copy()
        target = self.get_target(network)
        parent = network.get_parent(target)
        if target:
            replacement = self.factory(network.next_seg_id(), parent)[0]
            network.replace_node(target, replacement)

            return network, {'target': target, 'replacement': replacement}

        return None, {}


class TurnAngleMutator(SegmentReplacingMutator):
    NAME = 'turn_angle_mutator_{}'
    TURN_MIN = 15
    TURN_MAX = 90

    def __init__(self, rng, angle):
        super().__init__(TurnAngleMutator.NAME.format(angle), rng)
        self.angle = angle

    def get_target(self, network):
        turns = list()
        turns.extend(network.get_nodes(TYPE_L_TURN))
        turns.extend(network.get_nodes(TYPE_R_TURN))
        if turns:
            self.rng.shuffle(turns)
            for turn in turns:
                if turn.angle != self.angle:
                    return turn

        return None

    def get_replacement(self, network, target):
        parent = network.get_parent(target)
        factory = generate_turn_factory(
            self.angle, target.pivot_off, target.pivot_angle)
        replacement = factory(network.next_seg_id(), parent)[0]
        return replacement


class StraightLengthMutator(SegmentReplacingMutator):
    NAME = 'straight_length_mutator_{}'

    def __init__(self, rng, length):
        super().__init__(StraightLengthMutator.NAME.format(length), rng)
        self.length = length

    def get_target(self, network):
        straights = list()
        straights.extend(network.get_nodes(TYPE_STRAIGHT))
        if straights:
            self.rng.shuffle(straights)
            for straight in straights:
                if straight.length != self.length:
                    return straight

        return None

    def get_replacement(self, network, target):
        parent = network.get_parent(target)
        factory = generate_straight_factory(self.length)
        replacement = factory(network.next_seg_id(), parent)[0]
        return replacement
