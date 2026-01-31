import torch
import numpy as np
from torch.utils.data import Dataset
from DisentangledSSL.models import *

#############################
#  Synthetic Dataset Class  #
#############################

class MultimodalDataset(Dataset):
  def __init__(self, total_data, total_labels1 = None, total_labels2=None, total_labels3=None):
    self.data = torch.from_numpy(total_data).float()
    self.num_modalities = self.data.shape[0]
    self.targets1 = total_labels1
    self.targets2 = total_labels2
    self.targets3 = total_labels3
  
  def __len__(self):
    return self.data.shape[1]

  def __getitem__(self, idx):
    if self.targets1 is not None:
        return tuple([self.data[i, idx] for i in range(self.num_modalities)] + [self.targets1[idx]] + [self.targets2[idx]] + [self.targets3[idx]])
    else:
        return tuple([self.data[i, idx] for i in range(self.num_modalities)])
        
  def sample_batch(self, batch_size):
    sample_idxs = np.random.choice(self.__len__(), batch_size, replace=False)
    samples = self.__getitem__(sample_idxs)
    return samples


# Multiomic Dataset Class

class MultiomicDataset(Dataset):
    def __init__(self, total_data, total_labels1=None, total_labels2=None, total_labels3=None):
        # total_data is your unstacked list of modalities [X1, X2, ...]
        self.data = [torch.as_tensor(mod, dtype=torch.float32) for mod in total_data]
        self.num_samples = self.data[0].shape[0]
        self.num_modalities = len(self.data)
        assert all(m.shape[0] == self.num_samples for m in self.data), \
            "All modalities must have the same number of rows."

        # Normalize labels: turn each provided labels arg into a list of column tensors
        def _collect(arg):
            if arg is None:
                return []
            t = torch.as_tensor(arg, dtype=torch.float32) if not torch.is_tensor(arg) else arg.float()
            if t.ndim == 1:
                return [t]
            # reshape to (N, K) and split into columns
            t = t.reshape(t.shape[0], -1)
            return [t[:, j] for j in range(t.shape[1])]

        self.targets = []
        self.targets += _collect(total_labels1)
        self.targets += _collect(total_labels2)
        self.targets += _collect(total_labels3)

        for col in self.targets:
            assert col.shape[0] == self.num_samples, "Each label column must match number of samples."

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        modalities_data = [self.data[i][idx] for i in range(self.num_modalities)]
        if self.targets:
            label_parts = [col[idx] for col in self.targets]
            return tuple(modalities_data + label_parts)
        else:
            return tuple(modalities_data)

    def sample_batch(self, batch_size):
        idxs = np.random.choice(self.num_samples, batch_size, replace=False)
        return [self[i] for i in idxs]


def generate_data(n_samples, hidden_dim, dim_info, weight_info, label_weight_info1, label_weight_info2, label_weight_info3, seed=0, confounder=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = {}
    for k, d in {'Zs': dim_info['Zs'], 'Z1': dim_info['Z1'], 'Z2': dim_info['Z2']}.items():
        data[k] = np.random.multivariate_normal(np.zeros((d,)), np.eye(d)*0.5, size= n_samples)

    # Generate X by transforming Z1 and Zs
    t_Z1 = data['Z1'][:, :weight_info['Z1']]
    t_Zs = data['Zs'][:, :weight_info['Zs']]
    t_Z2 = data['Z2'][:, :weight_info['Z2']]

    if confounder:
      confounder = np.random.uniform(-2.0, 2.0, (t_Z1.shape[0], 1))

      confounder_object = {
          'confounder': confounder,
          'confounder_weight_Z1': 1,
          'confounder_weight_Zs': 1,
          'confounder_weight_Z2': 1,
      }
      
      t_Z1 = t_Z1 + confounder*confounder_object['confounder_weight_Z1']
      t_Zs = t_Zs + confounder*confounder_object['confounder_weight_Zs']
      t_Z2 = t_Z2 + confounder*confounder_object['confounder_weight_Z2']

      # t_Z1 = np.zeros_like(t_Z1) + confounder*confounder_object['confounder_weight_Z1']
      # t_Zs = np.zeros_like(t_Zs) + confounder*confounder_object['confounder_weight_Zs']
      # t_Z2 = np.zeros_like(t_Z2) + confounder*confounder_object['confounder_weight_Z2']
    else:
      confounder_object = {}

    Z = np.concatenate((t_Z1, t_Zs), axis=-1)
    T1 = np.random.uniform(-1.0,1.0,(Z.shape[-1], dim_info['X']))
    X = Z @ T1
    
    Z = np.concatenate((t_Z2, t_Zs), axis=-1)
    T2 = np.random.uniform(-1.0,1.0,(Z.shape[-1], dim_info['Y']))
    Y = Z @ T2
    total_data = [X, Y]  # keep modalities as a list with different column counts

    def get_label(label_weight_info, seed):
      p_Z1 = t_Z1[:, :label_weight_info['Z1']]
      p_Zs = t_Zs[:, :label_weight_info['Zs']]
      p_Z2 = t_Z2[:, :label_weight_info['Z2']]
      label_vector = np.concatenate((p_Z1, p_Zs, p_Z2), axis=-1) 
      torch.manual_seed(seed)
      label_mlp = mlp(label_vector.shape[1], hidden_dim = 100, output_dim=1, layers=2, activation='relu')
      for param in label_mlp.parameters():
          param.requires_grad = False
      label_vector = label_mlp(torch.Tensor(label_vector)).numpy() 
      label_vector = label_vector + np.random.normal(0, 0.1, label_vector.shape)
      label_prob = 1 / (1 + np.exp(-label_vector))
      midprob = np.median(label_prob)
      total_labels = (label_prob >= midprob).astype('float')
      total_labels = total_labels.reshape(-1)
      # total_labels = np.sum(label_vector, axis=1) > 0
      # total_labels = confounder > 0
      # total_labels = total_labels.reshape(-1)
      # total_labels = total_labels.astype('float')
      return total_labels
    
    total_labels1 = get_label(label_weight_info1, seed=0)
    total_labels2 = get_label(label_weight_info2, seed=1)
    total_labels3 = get_label(label_weight_info3, seed=2)
    
    print(total_data.shape, total_labels1.shape, total_labels2.shape, total_labels3.shape, X.shape, Y.shape, t_Z1.shape, t_Zs.shape, t_Z2.shape)
    return total_data, total_labels1, total_labels2, total_labels3, X, Y, t_Z1, t_Zs, t_Z2, confounder_object


##########################
#  Simple Augmentations  #
##########################

def noise(x, scale=0.01):
  noise = torch.randn(x.shape) * scale
  return x + noise.cuda()

def swap(x):
  mid = x.shape[0] // 2
  return torch.cat([x[mid:], x[:mid]])

def random_drop(x, drop_scale=10):
  drop_num = x.shape[0] // drop_scale
  drop_idxs = np.random.choice(x.shape[0], drop_num, replace=False)
  x_aug = torch.clone(x)
  x_aug[drop_idxs] = 0.0
  return x_aug

def identity(x):
  return x


# return augmented instance
def augment_data(x_batch, noise_scale=0.01, drop_scale=10):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = ['n', 'r', 'i']

  for i in range(x_batch.shape[0]):
    t_idxs = np.random.choice(3, 1, replace=False)
    t2 = transforms[t_idxs[0]]
    if t2 == 'n':
      v2[i] = noise(v2[i], scale=noise_scale)
    elif t2 == 'r':
      v2[i] = random_drop(v2[i], drop_scale=drop_scale)
    elif t2 == 'i':
       v2[i] = identity(v2[i])
  
  return v2
