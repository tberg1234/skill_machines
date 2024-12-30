"""
Q-Learning based method for learning skill primitives
"""

import random, time, numpy as np
from collections import defaultdict
from baselines import logger
from matplotlib import pyplot as plt


class SPEVF(object):
    def __init__(self, SP, primitive="g"):
        self.SP = SP
        self.primitives = list(self.SP.keys())
        self.primitive = primitive if primitive else self.primitives[0]
        self.evf = self.SP[self.primitive]
        self.goals = None
        self.goal = None
        self.initial_state = None

    @staticmethod
    def load(path):
        SP = np.load(path+".npy", allow_pickle=True).tolist()
        # print(SP["a"].keys())
        SP_ = {}
        for e in SP:
            EQ = SP[e]
            s = list(EQ.keys())[0]
            g = list(EQ[s].keys())[0]
            actions_n = len(EQ[s][g])
            EQ_ = defaultdict(lambda: defaultdict(lambda: np.zeros(actions_n)))
            for state in EQ:
                for goal in EQ[state]:
                    EQ_[state][goal] = EQ[state][goal]
            SP_[e] = EQ_

        return SPEVF(SP_)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def reset(self, **kwargs):
        self.goal = None
    
    def step(self, obs, **kwargs):
        obs = tuple(obs)
        if not self.goals:
            self.goals = list(self.evf[obs].keys())
        if not self.goal:
            self.goal = self._select_goal(obs)
        action = self.evf[obs][self.goal][:-1].argmax()
        print(obs, self.primitive, self.goal, self.evf[obs][self.goal][:-1], self.goals)
        return action, None, None, None

    def _select_goal(self, obs):
        values = [self.evf[obs][goal].max() for goal in self.goals]
        values = np.array(values)
        idx = np.random.choice(np.flatnonzero(values == values.max()))
        return self.goals[idx]

    def save(self, path, goals):
        SP = self.SP
        SP_ = {}
        for s in SP:
            SP_[s] = SP[s]
        np.save(path, (SP_,goals))


def load(path):
    """Load act function that was returned by learn function.
    Parameters
    ----------
    path: str
        path to the act function pickle
    Returns
    -------
    act: SPEVF
        function that takes a batch of observations
        and returns actions.
    """
    return SPEVF.load(path)

def load_sp(path,env,qinit):
    values_, env.sp_goals = np.load(path+".npy", allow_pickle=True).tolist()
    values = defaultdict(lambda: np.zeros(env.action_space.n)+qinit)
    for s in values_:
        values[s] = values_[s]
    return values


def get_best_action(Q,s,actions,q_init):
    best = [a for a in actions if Q[s][a] == Q[s].max()]
    return random.choice(best)

def learn(env,
          network=None,
          seed=None,
          lr=0.1,
          total_timesteps=100000,
          epsilon=0.1,
          print_freq=10000,
          gamma=0.9,
          q_init=0.0,
          sp=None,
          use_spe=False,
          use_csm=False,
          use_crm=False,
          use_rs=False,
          **others):
    """Train tabular skill primitives.
    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        This is just a placeholder to be consistent with the openai-baselines interface, but we don't really use state-approximation in tabular q-learning
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate
    total_timesteps: int
        number of env steps to optimizer for
    epsilon: float
        epsilon-greedy exploration
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    gamma: float
        discount factor
    q_init: float
        initial q-value for unseen states
    use_crm: bool
        use counterfactual experience to train the policy
    use_rs: bool
        use reward shaping
    """
    
    # Running Q-Learning
    reward_total = 0
    step = 0
    num_episodes = 0    
    
    rbarmin = env.rmin # extended reward penalty
    env.reset()
    SP = defaultdict(lambda: np.zeros(env.action_space.n)+q_init)
    actions = list(range(env.action_space.n))
    goals = set()

    while step < total_timesteps:
        s = tuple(env.reset())   
        s_, e_s, sp_s = env.sp_obs, env.get_events(), env.sp_state
        while True:
            # Selecting and executing the action
            a = random.choice(actions) if random.random() < epsilon else get_best_action(SP,s,actions,q_init)
            sn, r, done, _ = env.step(a)
            sn = tuple(sn)      
            sn_, e_sn, sp_sn = env.sp_obs, env.get_events(), env.sp_state

            # Updating the q-values
            experiences = [(s_,e_s,sp_s,a,r,sn_,e_sn,sp_sn,done)]
            for _s,_a,_r,_sn,_done in env.get_experiences(experiences):
                _s, _sn = tuple(_s), tuple(_sn)
                if _done: _delta = _r - SP[_s][_a]
                else:     _delta = _r + gamma*SP[_sn].max() - SP[_s][_a]
                SP[_s][_a] += lr*_delta
            
            # moving to the next state
            step += 1
            if step%print_freq == 0:
                SPEVF(SP).save(sp, env.sp_goals)
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("total reward", reward_total)
                logger.record_tabular("states", len(SP))
                logger.record_tabular("goals", len(env.sp_goals))
                logger.dump_tabular()
                reward_total = 0
            if done:
                reward_total += env.sp_reward
                num_episodes += 1
                break
            s = sn
            s_, e_s, sp_s = sn_, e_sn, sp_sn