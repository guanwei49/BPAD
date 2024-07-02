import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.utils.data as Data

from baseline.DRL.DDQN import DDQNAgent, train_ddqn
from baseline.DRL.VAEpre import PrewithVAE
from baseline.DRL.environment import Env


class DRL():
    def __init__(self, seed=None, hidden_dim=512, action_dim=2, gamma=0.99, lr=1e-3, batch_size=32,
                 memory_size=10000, num_episodes=15, steps=2000, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay=0.99):
        """
         Initializes the DRL class with the given parameters.

         Parameters:
         seed (int, optional): Random seed for reproducibility. Default is None.
         hidden_dim (int): Dimension of the hidden layers in the neural network. Default is 512.
         action_dim (int): Dimension of the action space. Default is 2.
         gamma (float): Discount factor for future rewards. Default is 0.99.
         lr (float): Learning rate for the optimizer. Default is 0.001.
         batch_size (int): Size of the batches used in training. Default is 32.
         memory_size (int): Capacity of the replay memory. Default is 10000.
         num_episodes (int): Number of episodes to train the model. Default is 15.
         steps (int): Number of steps per episode. Default is 2000.
         epsilon_start (float): Initial value of epsilon for the epsilon-greedy policy. Default is 1.0.
         epsilon_end (float): Final value of epsilon for the epsilon-greedy policy. Default is 0.1.
         epsilon_decay (float): Decay rate of epsilon per episode. Default is 0.99.
         """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'DRL'
        self.seed = seed
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.num_episodes = num_episodes
        self.steps = steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        if type(self.seed) is int:
            torch.manual_seed(self.seed)

    def fit(self, dataset):
        z_array, ri_array = PrewithVAE(dataset, self.device, n_epochs=500)
        # z_array, ri_array = PrewithVAE(dataset, self.device, n_epochs=100)

        # 创建环境
        env = Env(dataset, z_array, ri_array,self.steps)
        print("*" * 10 + "training DDQN" + "*" * 10)
        # 创建DDQN Agent
        self.agent = DDQNAgent(state_dim=dataset.attribute_dims[0], hidden_dim=self.hidden_dim,
                               action_dim=self.action_dim, trace_len=dataset.max_len,
                               device=self.device, env=env, gamma=self.gamma, lr=self.lr, batch_size=self.batch_size,
                               memory_size=self.memory_size)

        # 训练DDQN Agent
        train_ddqn(self.agent, env, num_episodes=self.num_episodes,  epsilon_start=self.epsilon_start,
                   epsilon_end=self.epsilon_end, epsilon_decay=self.epsilon_decay)

        return self

    def detect(self, dataset):
        Xs = torch.tensor(dataset.onehot_features[0])
        mask = torch.BoolTensor(dataset.mask)

        detect_dataset = Data.TensorDataset(Xs, mask)

        detect_dataloader = DataLoader(detect_dataset, batch_size=self.batch_size,
                                       shuffle=False, num_workers=0, pin_memory=True)

        model = self.agent.policy_net

        model.eval()

        with torch.no_grad():
            trace_level_abnormal_scores = []
            print("*" * 10 + "detecting" + "*" * 10)
            for batch_X, batch_mask in tqdm(detect_dataloader):
                batch_mask = batch_mask.to(self.device)
                batch_X = batch_X.to(self.device)

                # trace_level_abnormal_score = model(batch_X, batch_mask).max(1)[1].cpu()

                trace_level_abnormal_score = model(batch_X, batch_mask)[:,1] - model(batch_X, batch_mask)[:,0]

                trace_level_abnormal_scores.append(trace_level_abnormal_score)

            trace_level_abnormal_scores = torch.cat(trace_level_abnormal_scores, dim=0).detach().cpu()

            return trace_level_abnormal_scores, None, None
