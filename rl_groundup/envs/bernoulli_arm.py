import numpy as np

class BernoulliArm(object):
    def __init__(self, P):
        self.P = P

    def pull_arm(self):
        rand = np.random.random()
        return rand > self.P
