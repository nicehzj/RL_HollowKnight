import torch
import torch.optim as optim
import torch.nn.functional as F
from HKEnv import CustomEnv
from model import DQN
from buffer import ReplayBuffer

def train_dqn(num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
    env = CustomEnv()
    model = DQN(env.observation_space.shape[0], env.action_space.n)
    target_model = DQN(env.observation_space.shape[0], env.action_space.n)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters())
    buffer = ReplayBuffer(10000)
    
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = model(torch.FloatTensor(state)).argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                
                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
                state_action_values = model(torch.FloatTensor(states)).gather(1, torch.LongTensor(actions).unsqueeze(1))
                
                # Compute V(s_{t+1}) for all next states
                next_state_values = target_model(torch.FloatTensor(next_states)).max(1)[0].detach()
                
                # Compute the expected Q values
                expected_state_action_values = torch.FloatTensor(rewards) + (gamma * next_state_values * (1 - torch.FloatTensor(dones)))
                
                # Compute Huber loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())
        
        print(f"Episode {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    train_dqn(num_episodes=1000, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
