class PathEstimator:
    def score_path(self, polyline):
        raise NotImplementedError()


class RandomPathEstimator:

    def score_path(self, path, polyline):
        return random.uniform(0, 10)


class TurnAndLengthEstimator:
    def score_path(self, path, polyline):
        return score_path_polyline(polyline)

class LengthEstimator:
    def score_path(self, path, polyline):
        return len(path)
