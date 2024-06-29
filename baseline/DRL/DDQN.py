import math, random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from collections import deque

from tqdm import tqdm


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask, -1e-10)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)


        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class LSTMAttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, trace_len):
        super(LSTMAttentionNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = MultiHeadAttention(d_model=hidden_dim, n_head = 1)
        self.fc1 = nn.Linear(hidden_dim * trace_len, 2 * hidden_dim * trace_len)
        self.fc2 = nn.Linear(2 * hidden_dim * trace_len, output_dim)

    def forward(self, x, mask):
        lstm_out, _ = self.lstm(x)
        lstm_out = F.relu(lstm_out)

        mask_att = mask.unsqueeze(1).unsqueeze(1)
        attention_out = self.attention(q=lstm_out, k=lstm_out, v=lstm_out, mask=mask_att)

        flat = attention_out.masked_fill(mask.unsqueeze(2), 0).flatten(1, 2)

        x = F.relu(self.fc1(flat))
        x = self.fc2(x)

        return x



class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, mask, action, reward, next_state, next_mask, done):
        state = np.expand_dims(state, 0)
        mask = np.expand_dims(mask, 0)
        next_state = np.expand_dims(next_state, 0)
        next_mask = np.expand_dims(next_mask, 0)

        self.buffer.append((state, mask, action, reward, next_state, next_mask, done))

    def sample(self, batch_size):
        state, mask, action, reward, next_state, next_mask, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), np.concatenate(mask), action, reward, np.concatenate(next_state), np.concatenate(next_mask), done

    def __len__(self):
        return len(self.buffer)



class DDQNAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, trace_len, device, gamma=0.99, lr=1e-3, batch_size=32,
                 memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(memory_size)
        self.device = device

        self.policy_net = LSTMAttentionNetwork(state_dim, hidden_dim, action_dim,trace_len).to(self.device)
        self.target_net = LSTMAttentionNetwork(state_dim, hidden_dim, action_dim,trace_len).to(self.device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer= self.optimizer,
            gamma=0.95
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def store_transition(self, state, mask, action, reward, next_state, next_mask, done):
        self.replay_buffer.push(state, mask, action, reward, next_state, next_mask, done)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, mask, action, reward, next_state, next_mask, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        mask = torch.BoolTensor(mask).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        next_mask = torch.BoolTensor(np.float32(next_mask)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.policy_net(state, mask)
        next_q_values = self.policy_net(next_state, next_mask)
        next_q_state_values = self.target_net(next_state, next_mask)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.data).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def select_action(self,state, mask, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
            q_value = self.policy_net(state, mask)
            action = q_value.max(1)[1].cpu().data.numpy()[0]
        else:
            action = random.randrange(self.action_dim)
        return action


def train_ddqn(agent, env, num_episodes, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99):
    epsilon = epsilon_start

    total_steps = 0
    for episode in range(num_episodes):
        state, mask = env.reset()
        total_reward = 0
        for _ in tqdm(range(env.max_steps)):
            action = agent.select_action(state, mask, epsilon)
            next_state, next_mask, reward, done = env.step(action)
            agent.store_transition(state, mask, action, reward, next_state, next_mask, done)
            agent.train()
            state = next_state
            total_reward += reward
            total_steps += 1
            if total_steps % 50 == 0:
                epsilon = max(epsilon_end, epsilon*epsilon_decay)
                print(f'Epsilon: {epsilon}')
            # if total_steps % 1000 == 0:
            #     agent.update_target_network()
            if total_steps % 1000 == 0:
                agent.scheduler.step()
                print(f"Lr: {agent.optimizer.state_dict()['param_groups'][0]['lr']}")

        if episode % 5 == 0: #The target network is updated every five episodes.
            agent.update_target_network()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

