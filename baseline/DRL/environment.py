import numpy as np


def find_nearest_and_farthest(points, target):
    """
    找到与给定向量最近和最远的点
    :param points: 一堆向量，形状为 (n, d)，n 是向量的个数，d 是每个向量的维度
    :param target: 给定的单个向量，形状为 (d,)
    :return: 与target最近的点和最远的点index
    """
    # 计算所有向量到目标向量的欧氏距离
    distances = np.linalg.norm(points - target, axis=1)

    # 找到最小和最大距离的索引
    nearest_index = np.argmin(distances)
    farthest_index = np.argmax(distances)

    # 返回最远和最近的点
    return farthest_index, nearest_index

def ext_reward(action, label):
    if action==1 and label==1:
        ext = 1
    elif action==0 and label==0:
        ext = 0
    else:
        ext = -1

    return ext


class Env:
    def __init__(self, dataset, z_array, ri_array, max_step):
        self.state_array = dataset.onehot_features[0]
        self.mask_array = dataset.mask
        self.labeled_anomalies_index = dataset.labeled_indices
        self.unlabeled_index = np.delete(np.arange(dataset.num_cases), dataset.labeled_indices, 0)
        self.z_array = z_array
        self.ri_array = ri_array
        self.pre_state = None
        self.pre_z = None
        self.pre_ri = None
        self.steps = 0
        self.pre_label = None
        self.max_steps = max_step

    def reset(self):
        # 生成随机的初始状态
        index = np.random.choice(self.unlabeled_index)
        self.pre_state = self.state_array[index]
        self.pre_z = self.z_array[index]
        self.pre_ri = self.ri_array[index]
        mask = self.mask_array[index]
        self.pre_label = 0
        self.steps = 0
        return self.pre_state, mask

    def step(self, action):
        self.steps += 1

        reward = ext_reward(action,self.pre_label) + self.pre_ri  # 奖励

        # 按照概率从不同数组中采样
        if np.random.rand(1) < 0.5:  #sample from labeled anomalies
            index = np.random.choice(self.labeled_anomalies_index)
            next_state = self.state_array[index]
            next_z = self.z_array[index]
            next_ri = self.ri_array[index]
            next_mask = self.mask_array[index]
            self.pre_label = 1
        else:  #sample from unlabeled dataset
            sample_indexes = np.random.choice(self.unlabeled_index,size=(int(len(self.unlabeled_index)*0.15),)) #The used subset is 15% of Du.
            zs = self.z_array[sample_indexes]
            states = self.state_array[sample_indexes]
            ris = self.ri_array[sample_indexes]
            masks = self.mask_array[sample_indexes]
            ps_index = find_nearest_and_farthest(zs, self.pre_z)
            index = ps_index[action]
            next_state = states[index]
            next_ri = ris[index]
            next_z = zs[index]
            next_mask = masks[index]
            self.pre_label = 0

        self.pre_state = next_state
        self.pre_z = next_z
        self.pre_ri = next_ri

        done = self.steps >= self.max_steps
        # done = False

        return next_state, next_mask, reward, done