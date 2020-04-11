import numpy as np

class BernoulliArm(object):
    '''Objects of this class form a single arm in a K-armed bandit.
       Player has probability of P of winning when pulling the arm.'''
    def __init__(self, P):
        self.P = P

    def pull_arm(self):
        rand = np.random.random()
        return rand < self.P

class KArmedBandit(object):
    '''Objects of this class are K-armed bandits. Allows objects to pull
    a specified arm and recieve a reward based on probabilities unknown to
    the player/agent.'''
    def __init__(self, k):
        self.arm_probas = [round(np.random.uniform(0.1, 0.5), 2) for i in range(k)]
        self.bandits = [BernoulliArm(p) for p in self.arm_probas]

    def pull_arm(self, arm):
        return int(self.bandits[arm].pull_arm())
