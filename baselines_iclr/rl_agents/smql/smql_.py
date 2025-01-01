"""
Q-Learning based method for skill machines
"""

import random, time, numpy as np
from collections import defaultdict
from sympy.logic import boolalg
from sympy import sympify, Symbol
from baselines import logger
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


def load_sp(path,env,qinit):
    values_, env.sp_goals = np.load(path+".npy", allow_pickle=True).tolist()
    values = defaultdict(lambda: np.zeros(env.action_space.n)+qinit)
    for s in values_:
        values[s] = values_[s][:env.action_space.n]
    SP = EVF(env = env, primitive = "max")
    SP.values = values
    return SP

class EVF():
    def __init__(self, env=None, primitive=None):
        self.env = env
        self.primitive = primitive
        self.goals = self.env.sp_goals
        self.goal = None
        self.values = defaultdict(lambda: np.zeros(env.action_space.n)+10)
    
    def __call__(self, obs, goal=None):
        goal = goal if goal else self.goal
        obs = self.env.get_sp_obs(obs, self.env.sp_state, goal, self.primitive, False)
        obs = tuple(obs)

        return self.values[obs] 

    def reset(self, obs, **kwargs):
        self.goal = self._select_goal(obs)
    
    def step(self, obs, goal=None, **kwargs):
        goal = goal if goal else self.goal
        values = self(obs,goal)[:-1]
        action = values.argmax()
        return action, _, _, _

    def _select_goal(self, obs):
        values = [self(obs,goal).max() for goal in self.goals]
        values = np.array(values)
        idx = np.random.choice(np.flatnonzero(values == values.max()))
        return self.goals[idx]

class ComposedEVF(EVF):
    def __init__(self, evfs, action_space=None, max_evf=None, compose="or"):
        super().__init__(evfs[0].env)
        self.compose = compose
        self.max_evf = max_evf
        self.evfs = evfs
    
    def __call__(self, obs, goal=None):
        goal = goal if goal else self.goal
        qs = [self.evfs[i](obs, goal) for i in range(len(self.evfs))]
        qs = np.stack(tuple(qs), 0)
        if self.compose=="or":
            q = qs.max(0)
        elif self.compose=="and":
            q = qs.min(0)
        else: #not
            q = self.max_evf(obs, goal) - qs[0]
        return q

def select_goal(skill, obs):
    values = [skill(obs,goal).max() for goal in skill.goals]
    values = np.array(values)
    idx = np.random.choice(np.flatnonzero(values == values.max()))
    return skill.goals[idx]
    
def OR(evfs):
    return ComposedEVF(evfs, compose="or")

def AND(evfs):
    return ComposedEVF(evfs, compose="and")

def NOT(evf, max_evf):
    return ComposedEVF([evf], max_evf=max_evf, compose="not")

def exp_value(SP, exp):   
    evf =  None
    exp = sympify(exp)
    if exp:    
        def convert(exp):
            if type(exp) == Symbol:
                compound = EVF(env=SP.env, primitive = str(exp))
                compound.values = SP.values
            elif type(exp) == boolalg.Or:
                compound = convert(exp.args[0])
                for sub in exp.args[1:]:
                    compound = OR([compound, convert(sub)])
            elif type(exp) == boolalg.And:
                compound = convert(exp.args[0])
                for sub in exp.args[1:]:
                    compound = AND([compound, convert(sub)])
            else:
                compound = convert(exp.args[0])
                compound = NOT(compound, SP)
            return compound            
        evf = convert(exp)
    return evf

def get_qmax(Q,s,actions,q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])
    return max(Q[s].values())

def get_best_action(Q,actions,q_init):
    best = [a for a in actions if Q[a] == Q.max()]
    return random.choice(best)

def learn(env,
          network=None,
          seed=None,
          save_gif="",
          save_episode=0,
          lr=0.1,
          total_timesteps=100000,
          epsilon=0.1,
          print_freq=10000,
          gamma=0.9,
          q_init=0.0,
          init_eval=True,
          beta=0.9,
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
    q_init = 0.001
    learning_epsilon = epsilon
    init_eval = print_freq if init_eval else 0
    beta = beta if beta!=None else gamma
    reward_total = 0
    successes = 0
    step = 0
    num_episodes = 0    
    eval_reward_total = 0
    eval_step = 0 
    Q = defaultdict(lambda: np.zeros(env.action_space.n)+q_init)
    actions = list(range(env.action_space.n))
    
    # Loading and composing skills for skill machine
    SM = {}
    SP = load_sp(sp, env, q_init)
    for i in range(len(env.skill_machines)):
        sm = env.skill_machines[i]
        SM[i] = {}
        for u in sm:
            SM[i][u] = exp_value(SP, sm[u])

    trajectory = []
    while step < total_timesteps:        
        # Selecting and executing the action
        if num_episodes % 2 == 0:
            epsilon = learning_epsilon
        else:
            epsilon = 0.1
        
        # Render
        if save_gif and num_episodes==save_episode:
            epsilon = 0
            env.unwrapped.start_positions = [(10,3)]
                
        s = tuple(env.reset())        
        bs, u, rm = env.obs, env.current_u_id, env.current_rm_id
        skill = SM[rm][u]
        goal = select_goal(skill, bs)
        
        while True:
            # Render
            if save_gif and num_episodes==save_episode:
                    image = env.render()      
                    image = Image.fromarray(np.uint8(image))
                    image = image.convert("P",palette=Image.ADAPTIVE)
                    trajectory.append(image)   
                    trajectory[0].save(save_gif+'.gif', save_all=True, append_images=trajectory[1:], optimize=False, duration=10, loop=0)
                    print(env.sp_state, bs, goal, beta*Q[s], (1-beta)*skill(bs,goal), skill.primitive, rm, u, env.skill_machines[rm][u]) 
                    # plt.pause(0.0001)
            
            # Using skill for current rm state
            bs, u_, rm_ = env.obs, env.current_u_id, env.current_rm_id
            if u_ != u or rm_ != rm:
                u, rm = u_, rm_
                skill = SM[rm][u]
                goal = select_goal(skill, bs)
            
            if not use_csm:
                a = random.choice(actions) if random.random() < epsilon else get_best_action(skill(bs, goal),actions,q_init)
            else:
                Q_max = np.array([max([beta*Q[s][a], (1-beta)*skill(bs, goal)[a]]) for a in actions])
                a = random.choice(actions) if random.random() < epsilon else get_best_action(Q_max,actions,q_init)
            sn, r, done, info = env.step(a)
            sn = tuple(sn)

            # Updating the q-values
            experiences = []
            if use_crm:
                # Adding counterfactual experience (this will alrady include shaped rewards if use_rs=True)
                for _s,_a,_r,_sn,_done in info["crm-experience"]:
                    experiences.append((tuple(_s),_a,_r,tuple(_sn),_done))
            elif use_rs:
                # Include only the current experince but shape the reward
                experiences = [(s,a,info["rs-reward"],sn,done)]
            else:
                # Include only the current experience (standard q-learning)
                experiences = [(s,a,r,sn,done)]


            if num_episodes % 2 == 0 and step >= init_eval:
                for _s,_a,_r,_sn,_done in experiences:
                    if _done: _delta = _r - Q[_s][_a]
                    else:     _delta = _r + gamma*Q[_sn].max() - Q[_s][_a]
                    Q[_s][_a] += lr*_delta

            # moving to the next state
            if num_episodes % 2 == 1:
                reward_total += r
                successes += r>=1
            if num_episodes % 2 == 0:
                step += 1
                if step%print_freq == 0 or step == init_eval:
                    logger.record_tabular("steps", step)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("eval total reward", reward_total)
                    logger.record_tabular("eval successes", successes/(num_episodes*0.5))
                    logger.dump_tabular()
                    reward_total = 0
                    successes = 0
            if done:
                num_episodes += 1
                break
            s = sn
