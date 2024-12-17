import os, argparse, random, time, torch, numpy as np, gymnasium as gym, envs
from stable_baselines3.common.logger import configure
from sm import TaskPrimitive, SkillMachine, BaseAgent, evaluate
from rm import Task


class QLAgent(BaseAgent):
    """Q-learning Agent"""
    
    def __init__(self, name, env, SM=None, save_dir=None, load=False, lr=0.5, gamma=0.9, qinit=0):
        self.values, self.lr, self.gamma, self.SM, self.qinit = {}, lr, gamma, SM, qinit
        self.action_space, self.observation_space = env.action_space, env.observation_space
        if load: self.values = torch.load(save_dir+"wvf_"+name)
        
    def get_action_value(self, state):
        stateb = gym.spaces.flatten(self.observation_space, state).tobytes()
        if stateb not in self.values: self.values[stateb] = np.zeros(self.action_space.n)
        action, value = self.values[stateb].argmax(), self.values[stateb].max()
        if self.SM:
            (Q_action, Q_value), ((SM_action,), (SM_value,)) = (action, value), self.SM.get_action_value(state)
            idx = np.argmax([self.gamma*Q_value, (1-self.gamma)*SM_value])
            action, value = [Q_action, SM_action][idx], [Q_value, SM_value][idx]
        return np.array([action]), np.array([value])
        
    def get_values(self, state):
        state = gym.spaces.flatten(self.observation_space, state).tobytes()
        if state not in self.values: self.values[state] = np.zeros(self.action_space.n) + self.qinit
        return self.values[state]    
    
    def update_values(self, state, action, reward, state_, done):
        state = gym.spaces.flatten(self.observation_space, state).tobytes()
        if state not in self.values: self.values[state] = np.zeros(self.action_space.n) + self.qinit
        if done: self.values[state][action] += self.lr * (reward - self.values[state][action])
        else:    self.values[state][action] += self.lr * (reward + self.gamma*self.get_values(state_).max() - self.values[state][action])


def learn(primitive_env, task_env, total_steps, fewshot=False, q_dir="vf", sp_dir="wvfs", log_dir="logs", load=False, gamma=0.9, lr=0.5, epsilon=0.5, qinit=0, print_freq=1000, seed=None):  
    """Q-Learning based method for solving temporal logic tasks zeroshot or fewshot using Skill Machines"""

    # Initialise the World Value Functions for the min ("0") and max ("1") WVFs
    SP = {primitive: QLAgent(primitive, primitive_env, save_dir=sp_dir, load=load, lr=lr, gamma=gamma, qinit=qinit) for primitive in ['0','1']}
    if load or fewshot: primitive_env.goals.update(torch.load(sp_dir+"goals")) 
    # Skill machine over the learned skill primitives
    SM = SkillMachine(primitive_env, SP) 
    # Initialise task specific value function for fewshot learning
    if fewshot: Q = QLAgent("skill", task_env, SM=SM, lr=lr, gamma=gamma, qinit=qinit)

    # Start Training
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    reward_total, successes, eval_reward_total, step, eval_successes, num_episodes, start_time = 0, 0, 0, 0, 0, 1, time.time()
    while step < total_steps:
        if fewshot:
            state, info = task_env.reset(seed=seed)   
            SM.reset(task_env.rm, info["true_propositions"])
        else:
            state, info = primitive_env.reset(seed=seed) 
        
        while True:            
            # Selecting and executing the action
            if fewshot:
                if random.random() < epsilon: action = task_env.action_space.sample()
                else:                         action = Q.get_action_value(state)[0][0]
                state_, reward, done, truncated, info = task_env.step(action)
                SM.step(task_env.rm, info["true_propositions"])
            else:
                if random.random() < epsilon: action = primitive_env.environment.action_space.sample()
                else:                         action = random.choice([action for action in range(primitive_env.action_space.n) if SP["1"].get_values(state)[action] == SP["1"].get_values(state).max()])
                state_, reward, done, truncated, info = primitive_env.step(action)
            
            # Updating q-values
            if fewshot: 
                Q.update_values(state, action, reward, state_, done)
            elif not load:
                tp_state, tp_state_ = state.copy(), state_.copy()
                for primitive in SP:
                    primitive_env.primitive = primitive
                    for desired_goal in primitive_env.goals.values():
                        tp_state["desired_goal"], tp_state_["desired_goal"] = desired_goal, desired_goal
                        tp_reward = primitive_env.compute_reward(info["achieved_goal"].reshape(1,-1), desired_goal.reshape(1,-1), info["env_reward"], done)[0]
                        SP[primitive].update_values(tp_state, action, tp_reward, tp_state_, done)
            
            # logging and moving to the next state
            step += 1; state = state_
            if step%print_freq == 0:       
                if fewshot: torch.save(Q, q_dir+"skill") 
                elif not load: 
                    for primitive in SP: torch.save(SP[primitive].values, sp_dir+"wvf_"+primitive) 
                    torch.save(primitive_env.goals, sp_dir+"goals") 
                logger.record("steps", step); logger.record("episodes", num_episodes); 
                logger.record("total reward", reward_total); logger.record("successes", successes/num_episodes)
                logger.record("eval total reward", eval_reward_total); logger.record("eval successes", eval_successes/num_episodes)
                logger.record("goals", len(primitive_env.goals)); logger.record("time elapsed", time.time()-start_time); logger.dump(step)
                reward_total, successes, eval_reward_total, eval_successes, num_episodes, start_time = 0, 0, 0, 0, 0, time.time()
            if done or truncated:
                num_episodes += 1; reward_total += reward; successes += reward>=primitive_env.rmax
                if fewshot:    eval_reward, eval_success = evaluate(task_env, SM=SM, skill=Q, episodes=1, epsilon=0.0, gamma=1, max_episode_steps=1000, seed=seed)
                elif task_env: eval_reward, eval_success = evaluate(task_env, SM=SM, episodes=1, epsilon=0.0, gamma=1, max_episode_steps=1000, seed=seed)
                if task_env: eval_reward_total += eval_reward; eval_successes += eval_success
                break
    return SP

parser = argparse.ArgumentParser()
parser.add_argument("--env", help="The agent's environment, or the RM augmented environment for a predefined task distribution.", default='Office-v0')
parser.add_argument("--ltl", help="The ltl task the agent should solve. Ignored if --env is a task. E.g. (F (c & X (F o))) & (G~d)", default='')
parser.add_argument("--total_steps", help="Total training steps", type=int, default=100000)
parser.add_argument("--load", help="Load pretrained skill primitives", action='store_true', default=False)
parser.add_argument("--fewshot", help="Fewshot transfer", action='store_true', default=False)
parser.add_argument("--sp_dir", help="Directory where the learned skill primitives will be saved", default='')
parser.add_argument("--q_dir", help="Directory where the learned task specific skill will be saved", default='')
parser.add_argument("--log_dir", help="Directory where the results will be saved", default='')
parser.add_argument("--qinit", help="Q-values optimistic initialisation", type=float, default=0.001)
parser.add_argument("--seed", help="Random seed", type=int, default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    gym.logger.set_level(gym.logger.ERROR) 
    log_dir = args.log_dir if args.log_dir else './data/logs/sp_ql/' + args.env + "/"
    sp_dir = args.sp_dir if args.sp_dir else './data/sp_ql/' + args.env + "/"
    q_dir = args.q_dir if args.q_dir else './data/sm_ql/' + args.env + "/"
    os.makedirs(sp_dir, exist_ok=True); os.makedirs(q_dir, exist_ok=True); 
    if args.fewshot: assert ("Task" in args.env) or args.ltl, "A task distribution or ltl specification must be provided"
    
    # Initialise MDP for the primitives (primitive_env), and MDP for the given temporal logic task (task_env)
    if "Task" in args.env: 
        primitive_env = TaskPrimitive(gym.make(args.env).environment, max_episode_steps=1000)
        task_env = gym.make(args.env)
    else:                  
        primitive_env = TaskPrimitive(gym.make(args.env, max_episode_steps=1000))
        task_env = Task(gym.make(args.env), args.ltl) if args.ltl else None

    # Train primitive skill (skill_primitive), and optionally task specific skill fewshot
    skill_primitive = learn(primitive_env, task_env, args.total_steps, args.fewshot, q_dir, sp_dir, log_dir, args.load, qinit=args.qinit, seed=args.seed)
