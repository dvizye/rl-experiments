import random
import sys
import copy
import pickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
import numpy
import theano.tensor as T
import theano
from theano import function, shared

from deeplearning.mlp import MLP

##
n_hidden = 5
discount_factor = 0.9
learning_rate = 0.1
p_exploration = 1.0
##

# use double-precision for convenience
theano.config.floatX = 'float64'

def make_rlglue_action(action):
    a = Action()
    a.intArray = [int(action)]
    return a

class mlp_agent(Agent):

    def __init__(self):
        self.numpy_rng = numpy.random.RandomState(42)

        self.state_size = None
        self.action_size = 1

        self.state_bounds = None
        self.num_actions = 3

        self.mlp = None

        self.prev_state = None
        self.prev_action = None

        self.p_exploration = p_exploration

    def agent_init(self, spec):
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(spec)
        assert TaskSpec.valid, "TaskSpec could not be parsed: " + spec

        assert len(TaskSpec.getIntObservations())==0, "expecting continuous observations only"
        assert len(TaskSpec.getDoubleActions())==0, "expecting discrete actions only"
        self.state_size = len(TaskSpec.getDoubleObservations())
        self.action_size = len(TaskSpec.getIntActions())
        assert self.action_size == 1, 'expecting one action dimension only'
        
        specObs = TaskSpec.getDoubleObservations()
        self.state_bounds = []
        for i in xrange(0,self.state_size):
            self.state_bounds += [(specObs[i][0],specObs[i][1])]
        specAct = TaskSpec.getIntActions()
        self.action_bounds = []
        for i in xrange(0,self.action_size):
            self.action_bounds += [(specAct[i][0],specAct[i][1])]
        assert self.action_bounds[0][0]==0, 'action indices should start at 0'
        self.num_actions = self.action_bounds[0][1] - self.action_bounds[0][0] + 1

        print('compiling model...')
        self.x_state = T.vector('x_state')
        self.x_action = T.iscalar('x_action')
        self.x_action_onehot = T.eq(numpy.arange(self.num_actions), self.x_action)
        self.x = T.concatenate([self.x_state, self.x_action_onehot])

        self.mlp = MLP(rng=self.numpy_rng, input=self.x, n_in=self.state_size+self.num_actions, n_hidden=n_hidden, n_out=1)
        self.evaluate = function([self.x_state,self.x_action], self.mlp.output[0])
        self.q_update = self.compile_q_update()

    def compile_q_update(self):
        x_next_state = T.vector('x_next_state')

        reward = T.scalar('r')
        maxq = T.scalar('maxq')
        cost = T.sum(T.sqr(self.mlp.output[0] - (reward + discount_factor*maxq)))
        updates = []
        for p in self.mlp.params:
            updates.append((p, p - learning_rate*T.grad(cost, p)))
        return function([self.x_state,self.x_action,reward,maxq], cost, updates=updates)

    def max_action(self, state):
        """return best action according to current estimate of Q"""
        best = 0
        bestq = self.evaluate(state, best)
        for a in range(1,self.num_actions):
            q = self.evaluate(state, a)
            if q > bestq:
                best = a
                bestq = q
        return best, bestq

    def random_action(self):
        return random.choice(range(0,self.num_actions))

    # Observation -> Action
    def agent_start(self, observation):

        state = observation.doubleArray
        if random.random() < self.p_exploration:
            action = self.random_action()
        else:
            action, _ = self.max_action(state)

        self.prev_state = copy.deepcopy(state)
        self.prev_action = copy.deepcopy(action)

        return make_rlglue_action(action)

    # R * Observation -> Action
    def agent_step(self, reward, observation):
        state = observation.doubleArray
        max_action, max_q = self.max_action(state)

        self.q_update(self.prev_state, self.prev_action, reward, max_q)

        if random.random() < self.p_exploration:
            action = self.random_action()
        else:
            action = max_action

        self.prev_state = copy.deepcopy(state)
        self.prev_action = copy.deepcopy(action)

        return make_rlglue_action(action)

    # R -> ()
    def agent_end(self, reward):
        self.q_update(self.prev_state, self.prev_action, reward, 0)
        self.p_exploration *= 0.999
        print self.p_exploration
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        if message == 'episode over':
            # TODO: train network by NFQ
            pass

if __name__=="__main__":
    AgentLoader.loadAgent(mlp_agent())

