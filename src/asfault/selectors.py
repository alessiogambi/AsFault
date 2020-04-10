import random
import logging as l

# TODO Consider to replace this with DEAP implementation
class MateSelector:
    def select(self, suite, ignore=set()):
        raise NotImplementedError()


# TODO Consider to replace this with DEAP implementation
class RandomMateSelector(MateSelector):
    def __init__(self):
        pass

    def select(self, suite, ignore=set()):
        options = [resident for resident in suite if resident not in ignore]
        return random.sample(options, 1)[0]


# TODO Consider to replace this with DEAP implementation
class TournamentSelector(MateSelector):
    def __init__(self, tourney_size):
        self.tourney_size = tourney_size

    @DeprecationWarning
    def select(self, suite, ignore=set()):
        suite = [test for test in suite if test not in ignore]
        best = None
        for i in range(self.tourney_size):
            if suite:
                idx = random.randint(0, len(suite) - 1)
                resident = suite.pop(idx)
                l.debug('Selecting Resident: %s', str(resident))
                assert resident.score
                if not best or resident.score > best.score:
                    best = resident
        return best
