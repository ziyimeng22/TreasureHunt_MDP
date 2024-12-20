import numpy as np
import random
import time

# MDP Solver
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
        print(Vs)
        return Vs   # maximum expected value (over all actions) for a given state s

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
        print(V)
        return V   # optimal value function for all states

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
        print(policy)
        return policy   # optimal action policy for a given state s


# run simulation
class Environment(object):

    def __init__(self, stateSpace, goal, trap, block):
        self.stateSpace=stateSpace
        self.goal=goal
        self.trap=trap
        self.block=block

    def actionSpaceFunction(self, s):
        actions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }
        return actions.values()

    def transitionFunction(self, s, a, sPrime):
        x, y = s
        xPrime, yPrime = a
        sNext = (x+xPrime, y+yPrime)
        if sNext in block or sNext not in self.stateSpace:
            return 1 * s==sPrime
        else:
            return 1 * sNext==sPrime

    def rewardFunction(self, s, a, sPrime):
        if sPrime == self.goal:
            return 5
        elif sPrime == self.trap:
            return -10
        elif sPrime == self.block:
            return 0
        else:
            return -1
        
class Agent(object):
    def __init__(self, start):
        self.state=start

    def updateState(self, s, a):
        x, y = s
        dx, dy = a
        sPrime = (x+dx, y+dy) 
        return sPrime

class Simulation(object):
    
    def __init__(self, environment, agent, getPolicy):
        self.environment=environment
        self.agent=agent
        self.getPolicy=getPolicy

    def playGame(self, start):
        state = start
        print("Starting Grid-World MDP Game!")
        time.sleep(1)
        while state != goal:
            print(f"\nAgent at: {state}")
            policy = self.getPolicy(state)
            actions, probabilities = zip(*policy.items())
            action = random.choices(actions, probabilities)[0]
            nextState = self.agent.updateState(state, action)
            if state == trap: 
                print("\nAgent fell into a TRAP! 😭")
            state = nextState
            time.sleep(1)
        if state == goal:
            print(f"\nAgent at: {state}")   
            print("\nAgent reached the GOAL! 🎉")


if __name__ == "__main__":

    stateSpace = [
        (0,0), (0,1), (0,2),
        (1,0), (1,1), (1,2),
        (2,0), (2,1), (2,2)
    ]
    goal = (0,2)
    trap = (1,2)
    block = [(1,1), (0,1)]
    start = (0,0)

    environment = Environment(stateSpace, goal, trap, block)
    agent = Agent(start)

    gamma = 0.9
    theta = 1e-4
    roundingTolerance = 1e-4
    V = {s:0 for s in stateSpace}

    bellmanUpdate = BellmanUpdate(stateSpace, environment.actionSpaceFunction, environment.transitionFunction, environment.rewardFunction, gamma)
    valueIteration = ValueIteration(stateSpace, theta, bellmanUpdate)
    V = valueIteration(V)
    getPolicy = GetPolicy(stateSpace, environment.actionSpaceFunction, environment.transitionFunction, environment.rewardFunction, gamma, V, roundingTolerance)

    simulate = Simulation(environment, agent, getPolicy)
    simulate.playGame(start)