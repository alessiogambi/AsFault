import logging as l
import random

from asfault.network import *


class MetaMutator():
    """ This mutation operator includes all the single purpose mutators"""
    def init_mutators(self):
        for key, factory in SEG_FACTORIES.items():
            name = 'repl_' + key
            self.mutators[name] = SegmentReplacingMutator(self.rng, name, key, factory)

    def roll_mutation(self, resident):
        return random.random() <= c.ev.mut_chance

    def attempt_mutation(self, resident, mutator):
        epsilon = 0.25
        while True:
            try:
                mutated, aux = mutator.apply(resident)
                if mutated and mutated.complete_is_consistent():
                    test = self.test_from_network(mutated)
                    return mutated, aux
            except Exception as e:
                l.error('Exception while creating test from child: ')
                l.exception(e)
            failed = self.rng.random()
            if failed < epsilon:
                break

            epsilon *= 1.1

        return None, {}

    def mutate(self, resident):
        mutators = [*self.mutators.values()]
        random.shuffle(mutators)
        while mutators:
            mutator = mutators.pop()
            mutated, aux = self.attempt_mutation(resident, mutator)
            if not aux:
                aux = {}

            aux['type'] = mutator
            if mutated:
                test = self.test_from_network(mutated)
                l.info('Successfully applied mutation: %s', str(type(mutator)))
                return test, aux

        return None, {}



class Mutator:
    def __init__(self, name, rng):
        self.name = name
        random = rng

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
            return random.choice(options)
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
            random.shuffle(turns)
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
            random.shuffle(straights)
            for straight in straights:
                if straight.length != self.length:
                    return straight

        return None

    def get_replacement(self, network, target):
        parent = network.get_parent(target)
        factory = generate_straight_factory(self.length)
        replacement = factory(network.next_seg_id(), parent)[0]
        return replacement
