import random
import sys
import copy
import pickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

from deeplearning.DBN import DBN

class dbn_agent(Agent):
    numpy_rng = numpy.random.RandomState(42)

    state_size = 0
    action_size = 0

    state_bounds = []
    action_bounds = []

    dbn = None

    #lastAction=Action()
    #lastObservation=Observation()
    
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
        self.dbn = DBN(numpy_rng = self.numpy_rng,
                       n_ins = self.state_size+self.action_size,
                       hidden_layers_sizes = [5,5],
                       n_outs = 1)

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
        # TODO: learn
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, inMessage):
        return "I don't know how to respond to your message";

if __name__=="__main__":
    AgentLoader.loadAgent(dbn_agent())

