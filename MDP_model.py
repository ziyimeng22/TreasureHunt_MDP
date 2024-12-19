import numpy as np
import random
import time

# class BellmanUpdate
class BellmanUpdate(object):

    def __init__(self, stateSpace, actionSpaceFunction, transitionFunction, rewardFunction, gamma):
        self.stateSpace=stateSpace
        self.actionSpaceFunction=actionSpaceFunction
        self.transitionFunction=transitionFunction
        self.rewardFunction=rewardFunction
        self.gamma=gamma

    def __call__(self, s, V):
        Qs={a:sum([self.transitionFunction(s, a, sPrime)*(self.rewardFunction(s, a, sPrime)+self.gamma*V[sPrime]) for sPrime in self.stateSpace])\
            for a in self.actionSpaceFunction(s)}
        Vs=max(Qs.values())
        return Vs   # maximum expected value (over all actions) for a given state s

# class ValueIteration
class ValueIteration(object):

    def __init__(self, stateSpace, theta, bellmanUpdate):
        self.stateSpace=stateSpace
        self.theta=theta
        self.bellmanUpdate=bellmanUpdate

    def __call__(self, V):
        delta = np.inf
        while (delta > self.theta):
            delta = 0
            for s in self.stateSpace:
                v = V[s]
                V[s] = self.bellmanUpdate(s, V)
                delta = max(delta, abs(v - V[s]))
        return V   # optimal value function for all states

# class GetPolicy
class GetPolicy(object):

    def __init__(self, stateSpace, actionSpaceFunction, transitionFunction, rewardFunction, gamma, V, roundingTolerance):
        self.stateSpace=stateSpace
        self.actionSpaceFunction=actionSpaceFunction
        self.transitionFunction=transitionFunction
        self.rewardFunction=rewardFunction
        self.gamma=gamma
        self.V=V
        self.roundingTolerance=roundingTolerance

    def __call__(self, s):
        Qs={a:sum([self.transitionFunction(s, a, sPrime)*(self.rewardFunction(s, a, sPrime)+self.gamma*V[sPrime]) for sPrime in self.stateSpace])\
            for a in self.actionSpaceFunction(s)}        
        optimalActionList = [a for a in self.actionSpaceFunction(s) if abs(Qs[a]-max(Qs.values())) < self.roundingTolerance]
        policy={a: 1/(len(optimalActionList)) for a in optimalActionList}
        return policy   # optimal action policy for a given state s


# run simulation

stateSpace = [
    (0,0), (0,1), (0,2),
    (1,0), (1,1), (1,2),
    (2,0), (2,1), (2,2)
]
goal = (0,2)
trap = (1,2)
block = [(1,1)]
start = random.choice(stateSpace)
actions = ["up", "down", "left", "right"]

def actionSpaceFunction(s):
    actions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    return actions.values()

def transitionFunction(s, a, sPrime):
    x, y = s
    xPrime, yPrime = a
    sNext = (x+xPrime, y+yPrime)
    if sNext in block or sNext not in stateSpace:
        return 1 * (s==sPrime)
    else:
        return 1 * (sNext==sPrime)

def rewardFunction(s, a, sPrime):
    if sPrime == goal:
        return 100
    elif sPrime == trap:
        return -100
    elif sPrime == block:
        return 0
    else:
        return -1
    
gamma = 0.9
theta = 1e-4
roundingTolerance = 1e-4
V = {s:0 for s in stateSpace}

bellmanUpdate = BellmanUpdate(stateSpace, actionSpaceFunction, transitionFunction, rewardFunction, gamma)
valueIteration = ValueIteration(stateSpace, theta, bellmanUpdate)
V = valueIteration(V)
getPolicy = GetPolicy(stateSpace, actionSpaceFunction, transitionFunction, rewardFunction, gamma, V, roundingTolerance)
    
def playGame():
    state = start
    print("Starting Grid-World MDP Game!")
    time.sleep(1)
    while state != goal and state not in trap:
        print(f"\nAgent at: {state}")
        policy = getPolicy(state)
        actions, probabilities = zip(*policy.items())
        action = random.choices(actions, probabilities)[0]
        x, y = state
        dx, dy = action
        next_state = (x+dx, y+dy)
        state = next_state
        time.sleep(1)
    if state == goal:
        print("\nAgent reached the GOAL! 🎉")
    elif state in trap:
        print("\nAgent fell into a TRAP! 😭")       

playGame()