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

from deeplearning.mlp import MLP

##
n_hidden = 5
##

class mlp_agent(Agent):

    def __init__(self):
        self.numpy_rng = numpy.random.RandomState(42)

        self.state_size = 0
        self.action_size = 0

        self.state_bounds = []
        self.action_bounds = []

        self.mlp = None

        #self.lastAction=Action()
        #self.lastObservation=Observation()
    
    def agent_init(self, spec):
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(spec)
        assert TaskSpec.valid, "TaskSpec could not be parsed: " + spec

        assert len(TaskSpec.getIntObservations())==0, "expecting continuous observations only"
        assert len(TaskSpec.getDoubleActions())==0, "expecting discrete actions only"
        self.state_size = len(TaskSpec.getDoubleObservations())
        self.action_size = len(TaskSpec.getIntActions())
        
        specObs = TaskSpec.getDoubleObservations()
        self.state_bounds = []
        for i in xrange(0,self.state_size):
            self.state_bounds += [(specObs[i][0],specObs[i][1])]
        specAct = TaskSpec.getIntActions()
        self.action_bounds = []
        for i in xrange(0,self.action_size):
            self.action_bounds += [(specAct[i][0],specAct[i][1])]

        # construct network
        self.x_state = T.vector('x_state')
        self.x_action = T.vector('x_action')
        self.x = T.join(0, [self.x_state, self.x_action])
        
        self.mlp = MLP(rng=self.numpy_rng, input=self.x, n_in=self.state_size+self.action_size, n_hidden=n_hidden, n_out=1)
    
    def on_policy_action(self, state):
        # TODO: return best action according to current estimate of Q
        pass

    # Observation -> Action
    def agent_start(self, observation):
        # TODO: save this state and action
        return_action = Action()
        return_action.intArray = self.on_policy_action(observation.doubleArray)
        return return_action
        
        #lastAction=copy.deepcopy(returnAction)
        #lastObservation=copy.deepcopy(observation)

    # R * Observation -> Action
    def agent_step(self, reward, observation):
        return self.agent_start(observation)

    # R -> ()
    def agent_end(self, reward):
        # TODO: train network by NFQ
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, inMessage):
        return "I don't know how to respond to your message";

if __name__=="__main__":
    AgentLoader.loadAgent(mlp_agent())

