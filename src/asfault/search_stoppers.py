class SearchStopper:
    def stopping_condition_met(self, execution):
        return False


class NeverStopSearchStopper(SearchStopper):
    def stopping_condition_met(self, test):
        """Stopping condition is NEVER me"""
        return False


class StopAtObeSearchStopper(SearchStopper):
    def stopping_condition_met(self, test):
        """Stopping condition is met is there's at least one OBE in the test"""
        return test.execution.oobs > 0
