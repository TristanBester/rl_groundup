import numpy as np

class BernoulliArm(object):
    def __init__(self, P):
        self.P = P

    def pull_arm(self):
        rand = np.random.random()
        return rand < self.P

class KArmedBandit(object):
    def __init__(self, k):
        self.arm_probas = [round(np.random.uniform(0.1, 0.5), 2) for i in range(k)]
        self.bandits = [BernoulliArm(p) for p in self.arm_probas]

    def pull_arm(self, arm):
        return int(self.bandits[arm].pull_arm())
