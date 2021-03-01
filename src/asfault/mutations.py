import logging as l
import random

from asfault.network import *
from asfault.tests import test_from_network

class MetaMutator():

    """ This mutator applies the standard logic to mutate sequences and tries to mutate each element in the sequence with
    a probability of 1/size(sequence). Note that this works only for the case that one road corresponds to one test """

    mutators = dict()

    def __init__(self):
        self.init_mutators()


    """ This mutation operator includes all the single purpose mutators: replace segment, increase lenght, change angle """
    def init_mutators(self):
        # TODO It seems that factories generate instances of specific road types
        for key, factory in SEG_FACTORIES.items():
            name = 'repl_' + key
            self.mutators[name] = SegmentReplacingMutator(name, key, factory)

    @DeprecationWarning
    def roll_mutation(self, resident):
        return random.random() <= c.ev.mut_chance

    @DeprecationWarning
    def attempt_mutation(self, resident, mutator):
        epsilon = 0.25
        while True:
            try:
                mutated, aux = mutator.apply(resident)
                if mutated and mutated.complete_is_consistent():
                    return mutated, aux
            except Exception as e:
                l.error('Exception while creating test from child: ')
                l.exception(e)
            failed = random.random()

            if failed < epsilon:
                break
            else:
                l.info("Retry the mutation. Increase give up probability by 10%")
                epsilon *= 1.1

        return None, {}

    def mutate(self, individual):
        """ Mutation applies only to elements of the test, i.e., segments in the path """
        sequence_size = len(individual.path)
        mutation_probability = 1.0 / sequence_size

        # Individual is a test
        # But we mutate the underlying road network
        # Clone the network object
        mutated_network = individual.network.copy()
        has_mutation = False
        for segment in [segment for segment in individual.path if segment.roadtype not in GHOST_TYPES]:
            if random.random() < mutation_probability:
                l.debug("Mutation: segment %s selected for mutation", segment)
                # Randomly pick a factory to generate the replacement
                mutators = [*self.mutators.values()]
                random.shuffle(mutators)
                mutator = mutators.pop()
                del mutators

                # This generates a copy of the input segment
                mutated_network, aux = mutator.apply_to(segment, mutated_network)

                if mutated_network and mutated_network is not None:
                    has_mutation = True
                    l.info('Replaced %s with %s', aux['target'], aux['replacement'])
                else:
                    l.warning("Error while applying mutator %s", str(type(mutator)))
                    return None

        if not has_mutation:
            l.info('Original Individual was not mutated !')
            # TODO Should this return the same individual instead?
            return None

        # At this point we can try to create a test from the mutation
        if mutated_network.complete_is_consistent():
            l.info('Successfully mutated: Test#%s', individual.test_id)
            test = test_from_network(mutated_network)
            return test
        else:
            l.warning("Mutated individual is not valid")
            return None

    @DeprecationWarning
    def mutate_old(self, resident):

        mutators = [*self.mutators.values()]
        random.shuffle(mutators)

        while mutators:
            mutator = mutators.pop()
            mutated, aux = self.attempt_mutation(resident, mutator)
            if not aux:
                aux = {}

            aux['type'] = mutator
            if mutated:
                test = test_from_network(mutated)
                l.info('Successfully applied mutation: %s', str(type(mutator)))
                return test

        return None

class Mutator:
    def __init__(self, name):
        self.name = name

    def apply(self, resident):
        raise NotImplementedError()

# TODO Note that we use factories for mutators here to ensure we generate pieces of the expected shape.

class SegmentReplacingMutator(Mutator):
    """Replace one segment of the road network with another one"""
    def __init__(self, name, key, factory):
        super().__init__(name)
        self.key = key
        self.factory = factory

    def get_target(self, network):
        options = [option for option in network.nodes.values() if option.roadtype not in GHOST_TYPES]
        options = [option for option in options if option.key != self.key]
        if options:
            return random.choice(options)
        return None

    def apply_to(self, target, network):
        if not target:
            return None
        parent = network.get_parent(target)
        replacement = self.factory(network.next_seg_id(), parent)[0]
        network.replace_node(target, replacement)
        return network, {'target': target, 'replacement': replacement}

    def apply(self, resident):
        network = resident.network.copy()
        return self.apply_to(self.get_target(network), network)


# TODO Create also a TurnPivotMutator
class TurnAngleMutator(SegmentReplacingMutator):
    """ A Replacing mutator which replaces a turn with another turn which has a different angle. This might result into
    creating a turn that has the opposite direction..."""
    NAME = 'turn_angle_mutator_{}'
    TURN_MIN = 15
    TURN_MAX = 90

    def __init__(self, angle):
        super().__init__(TurnAngleMutator.NAME.format(angle))
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

    def apply_to(self, target, network):
        if not target:
            return None

        if target.roadtype != TYPE_L_TURN and target.roadtype != TYPE_R_TURN:
            l.info("Cannot mutate turn angle for segment %s", target)
            return None

        parent = network.get_parent(target)
        # TODO Not really sure what will this do
        factory = generate_turn_factory(self.angle, target.pivot_off, target.pivot_angle)
        replacement = factory(network.next_seg_id(), parent)[0]
        network.replace_node(target, replacement)
        return network, {'target': target, 'replacement': replacement}

    @DeprecationWarning
    def get_replacement(self, target, network):
        parent = network.get_parent(target)
        factory = generate_turn_factory(
            self.angle, target.pivot_off, target.pivot_angle)
        replacement = factory(network.next_seg_id(), parent)[0]
        return replacement


class StraightLengthMutator(SegmentReplacingMutator):
    NAME = 'straight_length_mutator_{}'

    def __init__(self, length):
        super().__init__(StraightLengthMutator.NAME.format(length))
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

    def apply_to(self, target, network):
        if not target:
            return None

        if target.roadtype != TYPE_STRAIGHT:
            l.info("Cannot mutate length for segment %s", target)
            return None

        parent = network.get_parent(target)
        # TODO Not really sure what will this do
        factory = generate_straight_factory(self.length)
        replacement = factory(network.next_seg_id(), parent)[0]
        network.replace_node(target, replacement)
        return network, {'target': target, 'replacement': replacement}

    @DeprecationWarning
    def get_replacement(self, network, target):
        parent = network.get_parent(target)
        factory = generate_straight_factory(self.length)
        replacement = factory(network.next_seg_id(), parent)[0]
        return replacement
