import pandas as pd

import torch.optim as optim
from DisentangledSSL.models import *
from DisentangledSSL.losses import *
import torch
import torch.nn as nn
from DisentangledSSL.dataset import augment_data
import DisentangledSSL.utils
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from torch.utils.data import DataLoader
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from torch.utils.data import DataLoader

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)


def HSIC(x, y, s_x=1.0, s_y=1.0):
    """Biased HSIC with Gaussian kernels. x: (n, dx), y: (n, dy)."""
    m, _ = x.shape
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)
    H = torch.eye(m, device=x.device) - (1.0/m) * torch.ones((m,m), device=x.device)
    return torch.trace(L @ H @ K @ H) / ((m-1)**2)


def mlp_head(dim_in, feat_dim):
    return nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )


class MVInfoMaxModel(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, layers=1, activation='relu', initialization = 'kaiming', distribution='normal', vmfkappa=1,
                 lr=1e-3, ratio=1, use_label=False, beta_start_value=1e-3, beta_end_value=5, beta_n_iterations=5000, beta_start_iteration=1000, split=50,
                 head='none', simclr=False, encoder1=None, encoder2=None):
        
        super().__init__()
        self.lr = lr
        self.ratio = ratio
        self.use_label = use_label
        self.iterations = 0
        self.split = split
        if beta_end_value > 0:
            self.beta_scheduler = utils.ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                    n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)
        self.beta_start_value = beta_start_value
        self.beta_end_value = beta_end_value
        self.distribution = distribution
        self.vmfkappa = vmfkappa
        self.simclr = simclr
        self.embed_dim = embed_dim

        if encoder1 is not None:
            self.encoder_x1 = encoder1
        else:
            self.encoder_x1 = mlp(x1_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)

        if encoder2 is not None:
            self.encoder_x2 = encoder2
        else:
            self.encoder_x2 = mlp(x2_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)

        if head == 'linear':
            self.head1 = nn.Linear(embed_dim, embed_dim)
            self.head2 = nn.Linear(embed_dim, embed_dim)
        elif head == 'mlp':
            self.head1 = mlp_head(embed_dim, embed_dim)
            self.head2 = mlp_head(embed_dim, embed_dim)
        elif head == 'none':
            self.head1 = nn.Identity()
            self.head2 = nn.Identity()
        else:
            raise NotImplementedError

        self.phead1 = ProbabilisticEncoder(self.head1, distribution=distribution, vmfkappa=vmfkappa)
        self.phead2 = ProbabilisticEncoder(self.head2, distribution=distribution, vmfkappa=vmfkappa)

        #critics
        self.critic = SupConLoss()
        
    def name(self):
        return 'Multiview IB'

    def forward(self, x1, x2):
        self.iterations += 1
        e1 = self.encoder_x1(x1)
        e2 = self.encoder_x2(x2)
        if self.simclr:
            mu1 = self.head1(e1)
            mu2 = self.head2(e2)
        else:
            p_z1_given_v1, mu1 = self.phead1(e1)
            p_z2_given_v2, mu2 = self.phead2(e2)
        
        if self.simclr:
            z1 = mu1
            z2 = mu2
        else:
            z1 = p_z1_given_v1.rsample()
            z2 = p_z2_given_v2.rsample()

        z1, z2 = nn.functional.normalize(z1, dim=-1), nn.functional.normalize(z2, dim=-1)
        concat_embed = torch.cat([z1.unsqueeze(dim=1), z2.unsqueeze(dim=1)], dim=1)
        joint_loss, loss_x, loss_y = self.critic(concat_embed)        
    
        if self.distribution == 'normal':
            skl = kl_divergence(mu1, mu2)
        elif self.distribution == 'vmf':
            skl = kl_vmf(mu1, mu2)

        if self.beta_end_value > 0:
            beta = self.beta_scheduler(self.iterations)
        else:
            beta = self.beta_start_value
        loss = joint_loss + beta * skl

        return loss, {'loss': loss.item(), 'clip': joint_loss.item(), 'skl': skl.item(), 'loss_x': loss_x.item(), 'loss_y': loss_y.item(), 'beta': beta}
        
    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        e1 = self.encoder_x1(x1)
        e2 = self.encoder_x2(x2)
        return torch.cat([e1, e2], dim=1)

    def get_separate_embeddings(self, x):
        x1, x2 = x[0], x[1]
        e1 = self.encoder_x1(x1)
        e2 = self.encoder_x2(x2)
        return e1, e2 # return separate embeddings for the disentanglement afterwards



def train(model, train_loader, optimizer, train_dataset, test_dataset, num_epoch=50, num_labels=3):
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
    for _iter in range(num_epoch):
        logs = {}
        logs.update({'Epoch': _iter})
        loss_meter = utils.AverageMeter('loss')
        clip_meter = utils.AverageMeter('clip')
        skl_meter = utils.AverageMeter('skl')
        loss_x_meter = utils.AverageMeter('loss_x')
        loss_y_meter = utils.AverageMeter('loss_y')
        beta_meter = utils.AverageMeter('beta')

        for i_batch, data_batch in enumerate(train_loader):
            model.train()
            x1 = data_batch[0].float().cuda()
            x2 = data_batch[1].float().cuda()
            x1 = augment_data(x1)
            x2 = augment_data(x2)
            loss, train_logs = model(x1, x2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(train_logs['loss'])
            clip_meter.update(train_logs['clip'])
            skl_meter.update(train_logs['skl'])
            loss_x_meter.update(train_logs['loss_x'])
            loss_y_meter.update(train_logs['loss_y'])
            beta_meter.update(train_logs['beta'])
            
            if _iter == 0 and i_batch == 0:
                logs.update({'loss': 0, 'clip': 0, 'skl': 0, 'loss_x': 0, 'loss_y': 0, 'beta': 0})
              #  logs.update({'Test Acc 1': 0, 'Test Acc 2': 0, 'Test Acc 3': 0})
                utils.print_row([i for i in logs.keys()], colwidth=12)
            
            if i_batch == len(train_loader)-1:
                logs.update({'loss': loss_meter.avg, 'clip': clip_meter.avg, 'skl': skl_meter.avg, 'loss_x': loss_x_meter.avg, 'loss_y': loss_y_meter.avg, 'beta': beta_meter.avg})
                
        # if _iter == num_epoch-1:
        #     test_acc = linearprobe(model, train_dataset, test_dataset, num_labels=num_labels)
        #     logs.update({f'Test Acc {i+1}': test_acc[i] for i in range(num_labels)})
        
        utils.print_row([logs[key] for key in logs.keys()], colwidth=12)        

        lr_scheduler.step()
    return logs


@torch.no_grad()
def linearprobe(model, train_dataset, test_dataset, num_labels):
    model.eval()

    # Create internal dataloaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    def collect_embeddings_and_labels(dataloader):
        all_embeds = []
        all_labels = []

        for batch in dataloader:
            if num_labels == 1:
                mnist_imgs, svhn_imgs, digits = batch
                mnist_imgs = mnist_imgs.cuda()
                svhn_imgs = svhn_imgs.cuda()
                embeds = model.get_embedding(torch.stack((mnist_imgs, svhn_imgs)))
                all_embeds.append(embeds.cpu())
                all_labels.append(digits)
            else:
                modality1, modality2, *labels = batch
                modalities = torch.stack((modality1.cuda(), modality2.cuda())).float()
                embeds = model.get_embedding(modalities)
                all_embeds.append(embeds.cpu())
                for i in range(num_labels):
                    if len(all_labels) <= i:
                        all_labels.append([])
                    all_labels[i].append(labels[i])

        all_embeds = torch.cat(all_embeds).numpy()

        if num_labels == 1:
            all_labels = torch.cat(all_labels).numpy()
        else:
            all_labels = [torch.cat(label).numpy() for label in all_labels]

        return all_embeds, all_labels

    # Collect train/test embeddings and labels
    train_embeds, train_labels = collect_embeddings_and_labels(train_loader)
    test_embeds, test_labels = collect_embeddings_and_labels(test_loader)

    # Train and score linear probe(s)
    res = []
    if num_labels == 1:
        clf = LogisticRegression(max_iter=200).fit(train_embeds, train_labels)
        score = clf.score(test_embeds, test_labels)
        res.append(score)
    else:
        for i in range(num_labels):
            print(train_embeds.shape, train_labels[i].shape)
            clf = LogisticRegression(max_iter=200).fit(train_embeds, train_labels[i])
            score = clf.score(test_embeds, test_labels[i])
            res.append(score)

    return res


# def train( model, train_loader, optimizer, train_dataset, test_dataset, num_epoch=50, num_labels=32):
#     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=num_epoch, eta_min=0.0, last_epoch=-1
#     )    

#     for _iter in range(num_epoch):
#         logs = {}
#         logs.update({'Epoch': _iter})
#         loss_meter = utils.AverageMeter('loss')
#         clip_meter = utils.AverageMeter('clip')
#         skl_meter = utils.AverageMeter('skl')
#         loss_x_meter = utils.AverageMeter('loss_x')
#         loss_y_meter = utils.AverageMeter('loss_y')
#         beta_meter = utils.AverageMeter('beta')

#         for i_batch, data_batch in enumerate(train_loader):
#             model.train()
#             x1 = data_batch[0].float().cuda()
#             x2 = data_batch[1].float().cuda()
#             x1 = augment_data(x1)
#             x2 = augment_data(x2)
#             loss, train_logs = model(x1, x2)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             loss_meter.update(train_logs['loss'])
#             clip_meter.update(train_logs['clip'])
#             skl_meter.update(train_logs['skl'])
#             loss_x_meter.update(train_logs['loss_x'])
#             loss_y_meter.update(train_logs['loss_y'])
#             beta_meter.update(train_logs['beta'])
            
#             if _iter == 0 and i_batch == 0:
#                 logs.update({'loss': 0, 'clip': 0, 'skl': 0, 'loss_x': 0, 'loss_y': 0, 'beta': 0})
#                 for i in range(num_labels):
#                     logs.update({f'R2 {i+1}': 0.0, f'RMSE {i+1}': 0.0})
#                 logs.update({"Avg R²": 0.0})
#                 utils.print_row(list(logs.keys()), colwidth=12)  # header

#             if i_batch == len(train_loader)-1:
#                 logs.update({'loss': loss_meter.avg, 'clip': clip_meter.avg, 'skl': skl_meter.avg, 'loss_x': loss_x_meter.avg, 'loss_y': loss_y_meter.avg, 'beta': beta_meter.avg})

#                 # print per-epoch progress (for all but the final epoch with probe)
#         if _iter == num_epoch-1:
#             probe = linearproberegression(model, train_dataset, test_dataset, num_labels=num_labels)
#             for i, (r2, rmse) in enumerate(probe):
#                 logs.update({f'R2 {i+1}': r2, f'RMSE {i+1}': rmse})
#             avg_r2 = (sum(r for r, _ in probe) / len(probe)) if len(probe) > 0 else 0.0
#             logs.update({"Avg R²": avg_r2})

#         utils.print_row([logs[key] for key in logs.keys()], colwidth=12)        



    #     lr_scheduler.step()

    # return logs




@torch.no_grad()
def linearproberegression(model, train_dataset, test_dataset, num_labels):
    """
    Linear probe (REGRESSION): fits LinearRegression from embeddings -> targets.
    Returns a list of (R2, RMSE) per label column.
    """
    model.eval()

    # internal loaders (no shuffle)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    def _maybe_stack(m1, m2):
        # stack only if shapes match; otherwise pass list
        return torch.stack((m1, m2)) if m1.shape[1:] == m2.shape[1:] else [m1, m2]

    def collect_embeddings_and_labels(dataloader):
        all_embeds = []
        if num_labels == 1:
            all_labels = []
        else:
            all_labels = [[] for _ in range(num_labels)]

        for batch in dataloader:
            if num_labels == 1:
                mnist_imgs, svhn_imgs, digits = batch
                mnist_imgs = mnist_imgs.cuda(non_blocking=True).float()
                svhn_imgs  = svhn_imgs.cuda(non_blocking=True).float()
                model_inp = _maybe_stack(mnist_imgs, svhn_imgs)
                embeds = model.get_embedding(model_inp)
                all_embeds.append(embeds.detach().cpu())
                all_labels.append(digits.detach().cpu())
            else:
                modality1, modality2, *labels = batch
                modality1 = modality1.cuda(non_blocking=True).float()
                modality2 = modality2.cuda(non_blocking=True).float()
                model_inp = _maybe_stack(modality1, modality2)
                embeds = model.get_embedding(model_inp)
                all_embeds.append(embeds.detach().cpu())
                for i in range(num_labels):
                    all_labels[i].append(labels[i].detach().cpu())

        all_embeds = torch.cat(all_embeds, dim=0).numpy()
        if num_labels == 1:
            all_labels = torch.cat(all_labels, dim=0).numpy()
        else:
            all_labels = [torch.cat(col, dim=0).numpy() for col in all_labels]
        return all_embeds, all_labels

    # collect train/test
    train_embeds, train_labels = collect_embeddings_and_labels(train_loader)
    test_embeds,  test_labels  = collect_embeddings_and_labels(test_loader)

    # regression probe: report (R2, RMSE)
    res = []
    if num_labels is not None:
        for i in range(num_labels):
            y_tr = np.asarray(train_labels[i]).reshape(-1)
            y_te = np.asarray(test_labels[i]).reshape(-1)
            reg = LinearRegression()
            reg.fit(train_embeds, y_tr)
            pred = reg.predict(test_embeds)

            r2   = r2_score(y_te, pred)
            rmse = np.sqrt(mean_squared_error(y_te, pred))
            res.append((r2, rmse))

    return res

@torch.no_grad()
def linearprobe(model, train_dataset, test_dataset, num_labels):
    model.eval()

    # Create internal dataloaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    def collect_embeddings_and_labels(dataloader):
        all_embeds = []
        all_labels = []

        for batch in dataloader:
            if num_labels == 1:
                mnist_imgs, svhn_imgs, digits = batch
                mnist_imgs = mnist_imgs.cuda()
                svhn_imgs = svhn_imgs.cuda()
                embeds = model.get_embedding(torch.stack((mnist_imgs, svhn_imgs)))
                all_embeds.append(embeds.cpu())
                all_labels.append(digits)
            else:
                modality1, modality2, *labels = batch
                m1 = modality1.cuda().float()
                m2 = modality2.cuda().float()
                if m1.shape[1:] == m2.shape[1:]:
                    model_inp = torch.stack((m1, m2), dim=0)
                else:
                    model_inp = [m1, m2]
                embeds = model.get_embedding(model_inp)
                
                all_embeds.append(embeds.cpu())
                for i in range(num_labels):
                    if len(all_labels) <= i:
                        all_labels.append([])
                    all_labels[i].append(labels[i])

        all_embeds = torch.cat(all_embeds).numpy()

        if num_labels == 1:
            all_labels = torch.cat(all_labels).numpy()
        else:
            all_labels = [torch.cat(label).numpy() for label in all_labels]

        return all_embeds, all_labels

    # Collect train/test embeddings and labels
    train_embeds, train_labels = collect_embeddings_and_labels(train_loader)
    test_embeds, test_labels = collect_embeddings_and_labels(test_loader)

    # Train and score linear probe(s)
    res = []
    if num_labels == 1:
        clf = LogisticRegression(max_iter=200).fit(train_embeds, train_labels)
        score = clf.score(test_embeds, test_labels)
        res.append(score)
    else:
        for i in range(num_labels):
            print(train_embeds.shape, train_labels[i].shape)
            clf = LogisticRegression(max_iter=200).fit(train_embeds, train_labels[i])
            score = clf.score(test_embeds, test_labels[i])
            res.append(score)

    return res



class DisenModel(nn.Module):
    def __init__(self, zsmodel, x1_dim, x2_dim, hidden_dim, embed_dim, zs_dim=50, layers=1, activation='relu', initialization = 'xavier', 
                 lr=1e-3, lmd_start_value=0.1, lmd_end_value=5, lmd_n_iterations=5000, lmd_start_iteration=1000, hsic_weight=0.0,
                 ortho_norm=True, condzs=True, proj=False, usezsx=True, apdzs=True, encoder1=None, encoder2=None, projection_x1=None, projection_x2=None):
        super().__init__()
        self.lr = lr
        self.ortho_norm = ortho_norm
        self.condzs = condzs
        self.proj = proj
        self.usezsx = usezsx
        self.apdzs = apdzs
        self.iterations = 0
        if lmd_end_value > 0:
            self.lmd_scheduler = utils.ExponentialScheduler(start_value=lmd_start_value, end_value=lmd_end_value,
                                                    n_iterations=lmd_n_iterations, start_iteration=lmd_start_iteration)
        self.lmd_start_value = lmd_start_value
        self.lmd_end_value = lmd_end_value
        self.hsic_weight = hsic_weight

        if encoder1 is not None:
            self.encoder_x1 = encoder1
        else:
            if self.condzs:
                self.encoder_x1 = mlp(x1_dim+zs_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
            else:
                self.encoder_x1 = mlp(x1_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)

        if encoder2 is not None:
            self.encoder_x2 = encoder2
        else:
            if self.condzs:
                self.encoder_x2 = mlp(x2_dim+zs_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
            else:
                self.encoder_x2 = mlp(x2_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        
        if self.proj=='mlp':
            self.projection_x1 = mlp(embed_dim*2, embed_dim*2, embed_dim*2, 1, activation, initialization = initialization)
            self.projection_x2 = mlp(embed_dim*2, embed_dim*2, embed_dim*2, 1, activation, initialization = initialization)
        elif self.proj=='linear':
            self.projection_x1 = nn.Linear(embed_dim*2, embed_dim*2)
            self.projection_x2 = nn.Linear(embed_dim*2, embed_dim*2)
        elif self.proj=='custom':
            self.projection_x1 = projection_x1
            self.projection_x2 = projection_x2
        else:
            self.projection_x1 = nn.Identity()
            self.projection_x2 = nn.Identity()

        self.critic = SupConLoss()

        self.zsmodel = zsmodel
        self.zsmodel.requires_grad = False
        self.embed_dim = embed_dim

    def forward(self, x1, x2, v1, v2):
        self.iterations += 1

        zsx1 = self.zsmodel.encoder_x1(x1).detach()
        zsx2 = self.zsmodel.encoder_x2(x2).detach()
        zsxv1 = self.zsmodel.encoder_x1(v1).detach()
        zsxv2 = self.zsmodel.encoder_x2(v2).detach()

        if self.condzs:
            z1x1 = self.encoder_x1(torch.cat([x1, zsx1], dim=1))
            z1xv1 = self.encoder_x1(torch.cat([v1, zsxv1], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, zsx2], dim=1))
            z2xv2 = self.encoder_x2(torch.cat([v2, zsxv2], dim=1))

            z1x1 = self.encoder_x1(torch.cat([x1, zsx1], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, zsx2], dim=1))
            # z1x1 = self.encoder_x1(x1, zsx1)
            # z1xv1 = self.encoder_x1(v1, zsxv1)
            # z2x2 = self.encoder_x2(x2, zsx2)
            # z2xv2 = self.encoder_x2(v2, zsxv2)
        else:
            z1x1 = self.encoder_x1(x1)
            z1xv1 = self.encoder_x1(v1)
            z2x2 = self.encoder_x2(x2)
            z2xv2 = self.encoder_x2(v2)

        if self.apdzs:
            if self.usezsx:
                zjointx1 = torch.cat([z1x1, zsx1], dim=1)
                zjointx2 = torch.cat([z2x2, zsx2], dim=1)
                zjointxv1 = torch.cat([z1xv1, zsxv1], dim=1)
                zjointxv2 = torch.cat([z2xv2, zsxv2], dim=1)
            else:
                zjointx1 = torch.cat([z1x1, zsx2], dim=1)
                zjointx2 = torch.cat([z2x2, zsx1], dim=1)
                zjointxv1 = torch.cat([z1xv1, zsxv2], dim=1)
                zjointxv2 = torch.cat([z2xv2, zsxv1], dim=1)

            zjointx1 = self.projection_x1(zjointx1)
            zjointx2 = self.projection_x2(zjointx2)
            zjointxv1 = self.projection_x1(zjointxv1)
            zjointxv2 = self.projection_x2(zjointxv2)

            zjointx1, zjointx2 = nn.functional.normalize(zjointx1, dim=-1), nn.functional.normalize(zjointx2, dim=-1)
            zjointxv1, zjointxv2 = nn.functional.normalize(zjointxv1, dim=-1), nn.functional.normalize(zjointxv2, dim=-1)
            concat_embed_x1 = torch.cat([zjointx1.unsqueeze(dim=1), zjointxv1.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([zjointx2.unsqueeze(dim=1), zjointxv2.unsqueeze(dim=1)], dim=1)
        else:
            z1x1_norm, z2x2_norm = nn.functional.normalize(z1x1, dim=-1), nn.functional.normalize(z2x2, dim=-1)
            z1xv1_norm, z2xv2_norm = nn.functional.normalize(z1xv1, dim=-1), nn.functional.normalize(z2xv2, dim=-1)
            concat_embed_x1 = torch.cat([z1x1_norm.unsqueeze(dim=1), z1xv1_norm.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([z2x2_norm.unsqueeze(dim=1), z2xv2_norm.unsqueeze(dim=1)], dim=1)

        specific_loss_x1, loss_x1, loss_y1 = self.critic(concat_embed_x1)
        specific_loss_x2, loss_x2, loss_y2 = self.critic(concat_embed_x2)

        loss_specific = specific_loss_x1 + specific_loss_x2

        if self.lmd_end_value > 0:
            lmd = self.lmd_scheduler(self.iterations)
        else:
            lmd = self.lmd_start_value

        loss_ortho = 0.5 * (ortho_loss(z1x1, zsx1, norm=self.ortho_norm) + ortho_loss(z2x2, zsx2, norm=self.ortho_norm)) + \
                    0.5 * (ortho_loss(z1xv1, zsxv1, norm=self.ortho_norm) + ortho_loss(z2xv2, zsxv2, norm=self.ortho_norm))
        
        if self.hsic_weight > 0.0:
            hsic_sp = 0.5 * (
                HSIC(z1x1, zsx1) + HSIC(z2x2, zsx2) +
                HSIC(z1xv1, zsxv1) + HSIC(z2xv2, zsxv2)
            )
        else:
            hsic_sp = torch.zeros((), device=x1.device)

        loss = loss_specific + lmd * loss_ortho + self.hsic_weight * hsic_sp

        return loss, {
            'loss': loss.item(),
            'specific': loss_specific.item(),
            'ortho': loss_ortho.item(),
            'hsic': hsic_sp.item(),
            'hsic_w': float(self.hsic_weight),
            'lmd': lmd
        }
            
    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        zsx1 = self.zsmodel.encoder_x1(x1).detach()
        zsx2 = self.zsmodel.encoder_x2(x2).detach()
        if self.condzs:
            z1x1 = self.encoder_x1(torch.cat([x1, zsx1], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, zsx2], dim=1))
            #z1x1 = self.encoder_x1(x1, zsx1)
            #z2x2 = self.encoder_x2(x2, zsx2)
        else:
            z1x1 = self.encoder_x1(x1)
            z2x2 = self.encoder_x2(x2)
        return z1x1, z2x2
    
    def get_three_embeddings(self, x):
        x1, x2 = x[0], x[1]
        zsx1 = self.zsmodel.encoder_x1(x1).detach()
        zsx2 = self.zsmodel.encoder_x2(x2).detach()
        z1x1 = self.encoder_x1(torch.cat([x1, zsx1], dim=1))
        z2x2 = self.encoder_x2(torch.cat([x2, zsx2], dim=1))
        # z1x1 = self.encoder_x1(x1, zsx1)
        # z2x2 = self.encoder_x2(x2, zsx2)
        return z1x1, z2x2, zsx1, zsx2

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor




def train_Disen(model, train_loader, optimizer, train_dataset, test_dataset, num_epoch=50,
                noise_scale=0.01, drop_scale=10, num_labels=1,
                Y_train=None, Y_val=None, val_dataset=None, probe_freq=None):
    """
    Train DisenModel with optional linear probing for hyperparameter selection.

    Args:
        Y_train, Y_val: If provided, run linear probing to evaluate representations
        val_dataset: Validation dataset for linear probing
        probe_freq: Run linear probe every N epochs (default: only at end)
    """
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)

    # Determine when to run linear probing
    do_probe = Y_train is not None and Y_val is not None and val_dataset is not None
    if probe_freq is None:
        probe_freq = num_epoch  # Only at end by default

    best_val_r2 = -float('inf')
    best_epoch = 0
    probe_history = []

    for _iter in range(num_epoch):
        logs = {}
        logs.update({'Epoch': _iter})
        loss_meter = utils.AverageMeter('loss')
        specific_meter = utils.AverageMeter('specific')
        ortho_meter = utils.AverageMeter('ortho')
        lmd_meter = utils.AverageMeter('lmd')
        hsic_meter = utils.AverageMeter('hsic')

        for i_batch, data_batch in enumerate(train_loader):
            model.train()
            x1 = data_batch[0].float().cuda()
            x2 = data_batch[1].float().cuda()
            x1 = augment_data(x1, noise_scale, drop_scale)
            x2 = augment_data(x2, noise_scale, drop_scale)
            v1 = augment_data(x1, noise_scale, drop_scale)
            v2 = augment_data(x2, noise_scale, drop_scale)
            loss, train_logs = model(x1, x2, v1, v2)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_meter.update(train_logs['loss'])
            specific_meter.update(train_logs['specific'])
            ortho_meter.update(train_logs['ortho'])
            lmd_meter.update(train_logs['lmd'])
            hsic_meter.update(train_logs['hsic'])

            if _iter == 0 and i_batch == 0:
                logs.update({'loss': 0, 'specific': 0, 'ortho': 0, 'lmd': 0, 'hsic': 0})
                if do_probe:
                    logs.update({'val_r2': 0})
                utils.print_row([i for i in logs.keys()], colwidth=12)

            if i_batch == len(train_loader)-1:
                logs.update({'loss': loss_meter.avg, 'specific': specific_meter.avg, 'ortho': ortho_meter.avg, 'lmd': lmd_meter.avg, 'hsic': hsic_meter.avg})

        lr_scheduler.step()

        # Linear probing for hyperparameter selection
        if do_probe and ((_iter + 1) % probe_freq == 0 or _iter == num_epoch - 1):
            probe_results = linearprobe_regression(
                model, train_dataset, val_dataset, Y_train, Y_val
            )
            val_r2 = probe_results.get("all", probe_results.get("zs1+zs2", {})).get("r2_val", 0)
            logs['val_r2'] = val_r2
            probe_history.append({'epoch': _iter, 'results': probe_results})

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_epoch = _iter

            # Print probe results
            print_linearprobe_results(probe_results, epoch=_iter)
        elif do_probe:
            logs['val_r2'] = 0  # Placeholder for epochs without probing

        utils.print_row([logs[key] for key in logs.keys()], colwidth=12)

    if do_probe:
        print(f"\n[Linear Probe Summary] Best Val R²={best_val_r2:.4f} at epoch {best_epoch}")

    return logs





@torch.no_grad()
def linearprobe_Disen(model, train_dataset, test_dataset, num_labels=3):
    model.eval()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    def collect_embeddings(dataloader):
        all_spe1, all_spe2 = [], []
        all_labels = [[] for _ in range(num_labels)]

        for batch in dataloader:
            modalities = batch[: -num_labels]
            labels = batch[-num_labels:]

            modalities = [m.cuda() for m in modalities]
            spe1, spe2 = model.get_embedding((modalities[0], modalities[1]))

            all_spe1.append(spe1.cpu())
            all_spe2.append(spe2.cpu())

            for i in range(num_labels):
                all_labels[i].append(labels[i].cpu())

        all_spe1 = torch.cat(all_spe1).numpy()
        all_spe2 = torch.cat(all_spe2).numpy()
        all_labels = [torch.cat(label).numpy() for label in all_labels]

        return all_spe1, all_spe2, all_labels

    # Collect embeddings and labels
    train_spe1, train_spe2, train_labels = collect_embeddings(train_loader)
    test_spe1, test_spe2, test_labels = collect_embeddings(test_loader)

    score_spe1 = []
    score_spe2 = []

    for i in range(num_labels):
        # Specific encoder 1
        clf1 = LogisticRegression(max_iter=200).fit(train_spe1, train_labels[i])
        score1 = clf1.score(test_spe1, test_labels[i])
        score_spe1.append(score1)

        # Specific encoder 2
        clf2 = LogisticRegression(max_iter=200).fit(train_spe2, train_labels[i])
        score2 = clf2.score(test_spe2, test_labels[i])
        score_spe2.append(score2)

    return tuple(score_spe1), tuple(score_spe2)


@torch.no_grad()
def linearprobe_regression(model, train_dataset, val_dataset, Y_train, Y_val):
    """
    Linear probing with Ridge regression for continuous targets (e.g., metabolites).
    Use during encoder training to monitor representation quality.

    Args:
        model: MVInfoMaxModel (Step 1) or DisenModel (Step 2)
        train_dataset, val_dataset: datasets with (X1, X2, ...)
        Y_train, Y_val: continuous targets [N, D] (metabolites)

    Returns:
        dict with R² and RMSE for different latent combinations
    """
    model.eval()

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Convert Y to numpy
    Y_tr_np = Y_train.cpu().numpy() if isinstance(Y_train, torch.Tensor) else Y_train
    Y_val_np = Y_val.cpu().numpy() if isinstance(Y_val, torch.Tensor) else Y_val

    def collect_embeddings(dataloader, model_type):
        """Collect embeddings based on model type."""
        embeds = {"zs1": [], "zs2": [], "zc1": [], "zc2": []}

        for batch in dataloader:
            x1 = batch[0].float().cuda()
            x2 = batch[1].float().cuda()

            if model_type == "disen":
                # DisenModel: get_three_embeddings returns (zc1, zc2, zs1, zs2)
                zc1, zc2, zs1, zs2 = model.get_three_embeddings([x1, x2])
                embeds["zs1"].append(zs1.cpu())
                embeds["zs2"].append(zs2.cpu())
                embeds["zc1"].append(zc1.cpu())
                embeds["zc2"].append(zc2.cpu())
            else:
                # MVInfoMaxModel: get_separate_embeddings returns (e1, e2)
                e1, e2 = model.get_separate_embeddings([x1, x2])
                embeds["zs1"].append(e1.cpu())
                embeds["zs2"].append(e2.cpu())

        # Concatenate
        for k in embeds:
            if embeds[k]:
                embeds[k] = torch.cat(embeds[k]).numpy()
            else:
                embeds[k] = None

        return embeds

    # Determine model type
    model_type = "disen" if hasattr(model, 'get_three_embeddings') else "shared"

    train_embeds = collect_embeddings(train_loader, model_type)
    val_embeds = collect_embeddings(val_loader, model_type)

    # Linear probing with Ridge regression
    results = {}

    # Define latent configurations to test
    if model_type == "disen":
        configs = [
            ("zs1", lambda e: e["zs1"]),
            ("zs2", lambda e: e["zs2"]),
            ("zc1", lambda e: e["zc1"]),
            ("zs1+zs2", lambda e: np.hstack([e["zs1"], e["zs2"]])),
            ("zs1+zc1", lambda e: np.hstack([e["zs1"], e["zc1"]])),
            ("all", lambda e: np.hstack([e["zs1"], e["zs2"], e["zc1"]])),
        ]
    else:
        configs = [
            ("zs1", lambda e: e["zs1"]),
            ("zs2", lambda e: e["zs2"]),
            ("zs1+zs2", lambda e: np.hstack([e["zs1"], e["zs2"]])),
        ]

    for name, get_z in configs:
        Z_tr = get_z(train_embeds)
        Z_val = get_z(val_embeds)

        if Z_tr is None:
            continue

        ridge = Ridge(alpha=1.0).fit(Z_tr, Y_tr_np)

        Y_pred_tr = ridge.predict(Z_tr)
        Y_pred_val = ridge.predict(Z_val)

        r2_tr = r2_score(Y_tr_np, Y_pred_tr, multioutput='variance_weighted')
        r2_val = r2_score(Y_val_np, Y_pred_val, multioutput='variance_weighted')
        mse_tr = mean_squared_error(Y_tr_np, Y_pred_tr)
        mse_val = mean_squared_error(Y_val_np, Y_pred_val)
        rmse_tr = np.sqrt(mse_tr)
        rmse_val = np.sqrt(mse_val)

        results[name] = {
            "r2_train": r2_tr, "r2_val": r2_val,
            "mse_train": mse_tr, "mse_val": mse_val,
            "rmse_train": rmse_tr, "rmse_val": rmse_val
        }

    return results


def print_linearprobe_results(results, epoch=None):
    """Pretty print linear probing results."""
    prefix = f"[Epoch {epoch}] " if epoch is not None else ""
    print(f"\n{prefix}Linear Probe (Ridge Regression) Results:")
    print("-" * 70)
    print(f"  {'Representation':15s} | {'MSE (val)':12s} | {'RMSE (val)':12s}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"  Y ~ {name:12s} | {metrics['mse_val']:12.6f} | {metrics['rmse_val']:12.6f}")
    print("-" * 70)


# def train_Disen(
#     model, train_loader, optimizer, train_dataset, test_dataset, num_epoch=50,
#     noise_scale=0.01, drop_scale=10, num_labels=32,metabolite_outcomes=None
# ):

#     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)

#     for _iter in range(num_epoch):
#         logs = {}
#         logs.update({'Epoch': _iter})
#         loss_meter = utils.AverageMeter('loss')
#         specific_meter = utils.AverageMeter('specific')
#         ortho_meter = utils.AverageMeter('ortho')
#         lmd_meter = utils.AverageMeter('lmd')
#         hsic_meter = utils.AverageMeter('hsic')

#         # ---------------------- BATCH LOOP ----------------------
#         for i_batch, data_batch in enumerate(train_loader):
#             model.train()
#             x1_raw = data_batch[0].float().cuda()
#             x2_raw = data_batch[1].float().cuda()

#             # two independent views from the same raw sample
#             x1a = augment_data(x1_raw, noise_scale, drop_scale)
#             v1  = augment_data(x1_raw, noise_scale, drop_scale)

#             x2a = augment_data(x2_raw, noise_scale, drop_scale)
#             v2  = augment_data(x2_raw, noise_scale, drop_scale)

#             loss, train_logs = model(x1a, x2a, v1, v2)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             loss_meter.update(train_logs['loss'])
#             specific_meter.update(train_logs['specific'])
#             ortho_meter.update(train_logs['ortho'])
#             lmd_meter.update(train_logs['lmd'])
#             hsic_meter.update(train_logs['hsic'])
#             # ------------------- END BATCH LOOP ---------------------

#             # ------------------- TEST PROBE -------------------
#         logs.update({
#             'loss':    loss_meter.avg,
#             'specific': specific_meter.avg,
#             'ortho':    ortho_meter.avg,
#             'lmd':      lmd_meter.avg,
#             'hsic':     hsic_meter.avg,
#         })

#         # ------------------- TEST PROBE (last epoch only) -------------------
#         if _iter == num_epoch - 1:
#             # ---- Linear probe (Ridge) ----
#             (r2_spe1, rmse_spe1, mae_spe1), (r2_spe2, rmse_spe2, mae_spe2) = \
#                 linearprobe_Disen_regression(
#                     model, train_dataset, test_dataset,
#                     num_labels=num_labels, batch_size=128, alpha=1.0
#                 )

#             # ---- Nonlinear probe (Random Forest) ----
#             r2rf_spe1, r2rf_spe2 = nonlinearprobe_Disen_regression_rf(
#                 model, train_dataset, test_dataset,
#                 num_labels=num_labels, batch_size=128
#             )

#             # ---- Per-metabolite logs ----
#             for i in range(num_labels):
#                 logs.update({
#                     f'R2_spe1 {i+1}': float(r2_spe1[i]),
#                     f'RMSE_spe1 {i+1}': float(rmse_spe1[i]),
#                     f'MAE_spe1 {i+1}': float(mae_spe1[i]),
#                     f'R2_spe2 {i+1}': float(r2_spe2[i]),
#                     f'RMSE_spe2 {i+1}': float(rmse_spe2[i]),
#                     f'MAE_spe2 {i+1}': float(mae_spe2[i]),
#                     f'R2RF_spe1 {i+1}': float(r2rf_spe1[i]),
#                     f'R2RF_spe2 {i+1}': float(r2rf_spe2[i]),
#                 })

#             # ---- Macro logs ----
#             logs.update({
#                 'Avg R² spe1': float(np.mean(r2_spe1)),
#                 'Avg R² spe2': float(np.mean(r2_spe2)),
#                 'Avg RMSE spe1': float(np.mean(rmse_spe1)),
#                 'Avg RMSE spe2': float(np.mean(rmse_spe2)),
#                 'Avg MAE spe1': float(np.mean(mae_spe1)),
#                 'Avg MAE spe2': float(np.mean(mae_spe2)),
#                 'Avg R²_RF spe1': float(np.mean(r2rf_spe1)),
#                 'Avg R²_RF spe2': float(np.mean(r2rf_spe2)),
#             })

#             # ---------- PLOTS ----------
#             r2_spe1_arr   = np.array(r2_spe1, dtype=float)
#             r2_spe2_arr   = np.array(r2_spe2, dtype=float)
#             rmse_spe1_arr = np.array(rmse_spe1, dtype=float)
#             rmse_spe2_arr = np.array(rmse_spe2, dtype=float)
#             mae_spe1_arr  = np.array(mae_spe1, dtype=float)
#             mae_spe2_arr  = np.array(mae_spe2, dtype=float)
#             r2rf_spe1_arr = np.array(r2rf_spe1, dtype=float)
#             r2rf_spe2_arr = np.array(r2rf_spe2, dtype=float)

#             macro_spe1_r2   = float(r2_spe1_arr.mean())
#             macro_spe2_r2   = float(r2_spe2_arr.mean())
#             macro_spe1_rmse = float(rmse_spe1_arr.mean())
#             macro_spe2_rmse = float(rmse_spe2_arr.mean())
#             macro_spe1_mae  = float(mae_spe1_arr.mean())
#             macro_spe2_mae  = float(mae_spe2_arr.mean())
#             macro_spe1_r2rf = float(r2rf_spe1_arr.mean())
#             macro_spe2_r2rf = float(r2rf_spe2_arr.mean())

#             df_Y  = pd.read_csv('/users/antonios/data/metabolites.known.tsv', sep="\t", index_col=0)
#             metabolite_outcomes = list(df_Y.columns)
#             metabolite_labels = metabolite_outcomes if (metabolite_outcomes is not None and len(metabolite_outcomes) == num_labels) \
#                                 else [f"M{i+1}" for i in range(num_labels)]

#             idx = np.arange(num_labels); w = 0.4

#             # --- R² (linear) ---
#             plt.figure(figsize=(max(8, 0.4 * num_labels + 4), 4.5))
#             plt.bar(idx - w/2, r2_spe1_arr, width=w, label=f"spe1 • Macro R² = {macro_spe1_r2:.3f}")
#             plt.bar(idx + w/2, r2_spe2_arr, width=w, label=f"spe2 • Macro R² = {macro_spe2_r2:.3f}")
#             plt.xticks(idx, metabolite_labels, rotation=60, ha='right')
#             plt.ylabel("R²"); ymin = min(-1.0, r2_spe1_arr.min(), r2_spe2_arr.min(), -0.1)
#             plt.ylim(ymin, 1.0); plt.axhline(0.0, ls="--", lw=1)
#             plt.legend(loc="best"); plt.tight_layout()
#             plt.savefig("linear_probe_disen_regression_per_metabolite_R2.png", dpi=300); plt.close()

#             # --- RMSE (linear) ---
#             plt.figure(figsize=(max(8, 0.4 * num_labels + 4), 4.5))
#             plt.bar(idx - w/2, rmse_spe1_arr, width=w, label=f"spe1 • Macro RMSE = {macro_spe1_rmse:.3f}")
#             plt.bar(idx + w/2, rmse_spe2_arr, width=w, label=f"spe2 • Macro RMSE = {macro_spe2_rmse:.3f}")
#             plt.xticks(idx, metabolite_labels, rotation=60, ha='right')
#             plt.ylabel("RMSE"); plt.legend(loc="best"); plt.tight_layout()
#             plt.savefig("linear_probe_disen_regression_per_metabolite_RMSE.png", dpi=300); plt.close()

#             # --- MAE (linear) ---
#             plt.figure(figsize=(max(8, 0.4 * num_labels + 4), 4.5))
#             plt.bar(idx - w/2, mae_spe1_arr, width=w, label=f"spe1 • Macro MAE = {macro_spe1_mae:.3f}")
#             plt.bar(idx + w/2, mae_spe2_arr, width=w, label=f"spe2 • Macro MAE = {macro_spe2_mae:.3f}")
#             plt.xticks(idx, metabolite_labels, rotation=60, ha='right')
#             plt.ylabel("MAE"); plt.legend(loc="best"); plt.tight_layout()
#             plt.savefig("linear_probe_disen_regression_per_metabolite_MAE.png", dpi=300); plt.close()

#             # --- R² (Random Forest) ---
#             plt.figure(figsize=(max(8, 0.4 * num_labels + 4), 4.5))
#             plt.bar(idx - w/2, r2rf_spe1_arr, width=w, label=f"spe1 • Macro RF R² = {macro_spe1_r2rf:.3f}")
#             plt.bar(idx + w/2, r2rf_spe2_arr, width=w, label=f"spe2 • Macro RF R² = {macro_spe2_r2rf:.3f}")
#             plt.xticks(idx, metabolite_labels, rotation=60, ha='right')
#             plt.ylabel("R² (Random Forest)")
#             ymin_rf = min(-1.0, r2rf_spe1_arr.min(), r2rf_spe2_arr.min(), -0.1)
#             plt.ylim(ymin_rf, 1.0); plt.axhline(0.0, ls="--", lw=1)
#             plt.legend(loc="best"); plt.tight_layout()
#             plt.savefig("linear_probe_disen_regression_per_metabolite_R2_RF.png", dpi=300); plt.close()
#         # --------------------------------------------------------------------

#         lr_scheduler.step()
#         utils.print_row([logs[key] for key in logs.keys()], colwidth=12)

#     return logs



############## Linear probe for regression #####################
@torch.no_grad()
def linearprobe_Disen_regression(model, train_dataset, test_dataset, num_labels=32, batch_size=32, alpha=1.0):
    from torch.utils.data import DataLoader
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    model.eval()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,  shuffle=False)

    def collect_embeddings(dataloader):
        all_spe1, all_spe2 = [], []
        all_targets = [[] for _ in range(num_labels)]
        for batch in dataloader:
            modalities = batch[: -num_labels]
            targets    = batch[-num_labels:]
            modalities = [m.cuda() for m in modalities]
            spe1, spe2 = model.get_embedding(modalities)
            all_spe1.append(spe1.cpu()); all_spe2.append(spe2.cpu())
            for i in range(num_labels):
                all_targets[i].append(targets[i].cpu())
        X1 = torch.cat(all_spe1).numpy()
        X2 = torch.cat(all_spe2).numpy()
        Y  = [torch.cat(t).numpy().astype(np.float32) for t in all_targets]
        return X1, X2, Y

    X1_tr, X2_tr, Y_tr = collect_embeddings(train_loader)
    X1_te, X2_te, Y_te = collect_embeddings(test_loader)

    r2_spe1, rmse_spe1, mae_spe1 = [], [], []
    r2_spe2, rmse_spe2, mae_spe2 = [], [], []

    for i in range(num_labels):
        y_tr = Y_tr[i].reshape(-1)
        y_te = Y_te[i].reshape(-1)

        reg1 = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        reg1.fit(X1_tr, y_tr)
        yhat1 = reg1.predict(X1_te)
        r2_spe1.append(float(r2_score(y_te, yhat1)))
        rmse_spe1.append(float(np.sqrt(mean_squared_error(y_te, yhat1))))
        mae_spe1.append(float(mean_absolute_error(y_te, yhat1)))

        reg2 = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        reg2.fit(X2_tr, y_tr)
        yhat2 = reg2.predict(X2_te)
        r2_spe2.append(float(r2_score(y_te, yhat2)))
        rmse_spe2.append(float(np.sqrt(mean_squared_error(y_te, yhat2))))
        mae_spe2.append(float(mean_absolute_error(y_te, yhat2)))

    return (r2_spe1, rmse_spe1, mae_spe1), (r2_spe2, rmse_spe2, mae_spe2)

@torch.no_grad()
def nonlinearprobe_Disen_regression_rf(model, train_dataset, test_dataset, num_labels=32, batch_size=32):
    from torch.utils.data import DataLoader
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    model.eval()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,  shuffle=False)

    def collect_embeddings(dataloader):
        all_spe1, all_spe2 = [], []
        all_targets = [[] for _ in range(num_labels)]
        for batch in dataloader:
            modalities = batch[: -num_labels]
            targets    = batch[-num_labels:]
            modalities = [m.cuda() for m in modalities]
            spe1, spe2 = model.get_embedding(modalities)
            all_spe1.append(spe1.cpu()); all_spe2.append(spe2.cpu())
            for i in range(num_labels):
                all_targets[i].append(targets[i].cpu())
        X1 = torch.cat(all_spe1).numpy()
        X2 = torch.cat(all_spe2).numpy()
        Y  = [torch.cat(t).numpy().astype(np.float32) for t in all_targets]
        return X1, X2, Y

    X1_tr, X2_tr, Y_tr = collect_embeddings(train_loader)
    X1_te, X2_te, Y_te = collect_embeddings(test_loader)

    r2_rf_spe1, r2_rf_spe2 = [], []

    for i in range(num_labels):
        y_tr = Y_tr[i].reshape(-1)
        y_te = Y_te[i].reshape(-1)

        rf1 = RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            max_features='sqrt', random_state=0, n_jobs=-1
        ).fit(X1_tr, y_tr)
        rf2 = RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            max_features='sqrt', random_state=0, n_jobs=-1
        ).fit(X2_tr, y_tr)

        r2_rf_spe1.append(float(r2_score(y_te, rf1.predict(X1_te))))
        r2_rf_spe2.append(float(r2_score(y_te, rf2.predict(X2_te))))

    return r2_rf_spe1, r2_rf_spe2


################## Linear probe for logistic regression #####################

@torch.no_grad()
def linearprobe_Disen(model, train_dataset, test_dataset, num_labels=3):
    model.eval()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    def _to_model_input(mods):
        # If all modalities share the same trailing shape, stack to (M,B,d,...).
        # Otherwise, return the list and let the model handle unstacked inputs.
        try:
            same_shape = all(tuple(m.shape[1:]) == tuple(mods[0].shape[1:]) for m in mods)
            if same_shape:
                return torch.stack(mods, dim=0)
        except Exception:
            pass
        return mods  # unstacked list

    def collect_embeddings(dataloader):
        all_spe1, all_spe2 = [], []
        all_labels_lists = None  # will init on first batch

        for batch in dataloader:
            # Split batch elements by dimensionality:
            #   - modalities: tensors with dim >= 2 (e.g., [B, d_i])
            #   - labels:    tensors with dim == 1 (e.g., [B])
            items = list(batch)
            modalities = [t for t in items if t.dim() >= 2]
            labels     = [t for t in items if t.dim() == 1]

            # Infer num_labels from the batch if caller's value is wrong
            # (keeps compatibility with old calls that pass num_labels=3)
            if all_labels_lists is None:
                # initialize per-label collectors
                nlab = len(labels) if num_labels is None else len(labels)
                all_labels_lists = [[] for _ in range(nlab)]

            # Send modalities to CUDA
            modalities = [m.cuda(non_blocking=True) for m in modalities]

            # Prepare model input (stack only if shapes match)
            model_inp = _to_model_input(modalities)

            # Get embeddings (model must accept either stacked tensor or list)
            spe1, spe2 = model.get_embedding(model_inp)

            all_spe1.append(spe1.detach().cpu())
            all_spe2.append(spe2.detach().cpu())

            # Accumulate labels
            for i, lab in enumerate(labels):
                all_labels_lists[i].append(lab.detach().cpu())

        all_spe1 = torch.cat(all_spe1, dim=0).numpy()
        all_spe2 = torch.cat(all_spe2, dim=0).numpy()
        all_labels = [torch.cat(buf, dim=0).numpy() for buf in all_labels_lists]

        return all_spe1, all_spe2, all_labels

    # Collect embeddings and labels
    train_spe1, train_spe2, train_labels = collect_embeddings(train_loader)
    test_spe1,  test_spe2,  test_labels  = collect_embeddings(test_loader)

    # Use however many label columns came in
    K = len(train_labels)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np

    score_spe1, score_spe2 = [], []

    for i in range(len(train_labels)):
        # Specific encoder 1
        reg1 = LinearRegression().fit(train_spe1, train_labels[i])
        pred1 = reg1.predict(test_spe1)
        r2_1 = r2_score(test_labels[i], pred1)
        rmse_1 = np.sqrt(mean_squared_error(test_labels[i], pred1))
        score_spe1.append((r2_1, rmse_1))

        # Specific encoder 2
        reg2 = LinearRegression().fit(train_spe2, train_labels[i])
        pred2 = reg2.predict(test_spe2)
        r2_2 = r2_score(test_labels[i], pred2)
        rmse_2 = np.sqrt(mean_squared_error(test_labels[i], pred2))
        score_spe2.append((r2_2, rmse_2))

    return tuple(score_spe1), tuple(score_spe2)




class JointDisenModel(nn.Module):
    """
    Joint optimization:
    max I(zsx;Y) + a * [I(zsx,z1x;X) - lmd*I(zsx;z1x)]
    zsx has the probablistic encoder; z1x has the deterministic encoder
    """
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, layers=2, activation='relu', initialization='xavier', 
                 distribution='normal', vmfkappa=1, lr=1e-4,
                 lmd_start_value=1e-3, lmd_end_value=1, lmd_n_iterations=100000, lmd_start_iteration=50000,
                 a=1, ortho_norm=True, condzs=True, proj=False, usezsx=True, apdzs=True):
        super().__init__()
        self.lr = lr
        self.ortho_norm = ortho_norm
        self.condzs = condzs
        self.proj = proj
        self.usezsx = usezsx
        self.apdzs = apdzs
        self.vmfkappa = vmfkappa
        self.iterations = 0
        if lmd_end_value > 0:
            self.lmd_scheduler = utils.ExponentialScheduler(start_value=lmd_start_value, end_value=lmd_end_value,
                                                    n_iterations=lmd_n_iterations, start_iteration=lmd_start_iteration)
        self.lmd_start_value = lmd_start_value
        self.lmd_end_value = lmd_end_value
        self.a = a

        self.encoder_x1s = mlp(x1_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        self.encoder_x2s = mlp(x2_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)

        self.phead1 = ProbabilisticEncoder(nn.Identity(), distribution=distribution, vmfkappa=vmfkappa)
        self.phead2 = ProbabilisticEncoder(nn.Identity(), distribution=distribution, vmfkappa=vmfkappa)

        if self.condzs:
            self.encoder_x1 = mlp(x1_dim+embed_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
            self.encoder_x2 = mlp(x2_dim+embed_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        else:
            self.encoder_x1 = mlp(x1_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
            self.encoder_x2 = mlp(x2_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)

        if self.proj:
            self.projection_x1 = mlp(embed_dim*2, embed_dim*2, embed_dim*2, 1, activation, initialization = initialization)
            self.projection_x2 = mlp(embed_dim*2, embed_dim*2, embed_dim*2, 1, activation, initialization = initialization)

        self.critic = SupConLoss()
        self.embed_dim = embed_dim

    def forward(self, x1, x2, v1, v2):
        self.iterations += 1
        e1 = self.encoder_x1s(x1)
        e2 = self.encoder_x2s(x2)
        e1_v = self.encoder_x1s(v1)
        e2_v = self.encoder_x2s(v2)

        p_zs1_given_x1, mu1 = self.phead1(e1)
        p_zs2_given_x2, mu2 = self.phead2(e2)
        p_zsv1_given_v1, mu1_v = self.phead1(e1_v)
        p_zsv2_given_v2, mu2_v = self.phead2(e2_v)

        zs1 = p_zs1_given_x1.rsample()
        zs2 = p_zs2_given_x2.rsample()
        zsv1 = p_zsv1_given_v1.rsample()
        zsv2 = p_zsv2_given_v2.rsample()

        concat_embed = torch.cat([zs1.unsqueeze(dim=1), zs2.unsqueeze(dim=1)], dim=1)
        concat_embed_v = torch.cat([zsv1.unsqueeze(dim=1), zsv2.unsqueeze(dim=1)], dim=1)
        joint_loss, loss_x, loss_y = self.critic(concat_embed)
        joint_loss_v, loss_x_v, loss_y_v = self.critic(concat_embed_v)
        joint_loss = 0.5 * (joint_loss + joint_loss_v)
        loss_x = 0.5 * (loss_x + loss_x_v)
        loss_y = 0.5 * (loss_y + loss_y_v)
        loss_shared = joint_loss

        if self.condzs:
            z1x1 = self.encoder_x1(torch.cat([x1, e1], dim=1))
            z1xv1 = self.encoder_x1(torch.cat([v1, e1_v], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, e2], dim=1))
            z2xv2 = self.encoder_x2(torch.cat([v2, e2_v], dim=1))
        else:
            z1x1 = self.encoder_x1(x1)
            z1xv1 = self.encoder_x1(v1)
            z2x2 = self.encoder_x2(x2)
            z2xv2 = self.encoder_x2(v2)

        if self.apdzs:
            if self.usezsx:
                zjointx1 = torch.cat([z1x1, e1], dim=1)
                zjointx2 = torch.cat([z2x2, e2], dim=1)
                zjointxv1 = torch.cat([z1xv1, e1_v], dim=1)
                zjointxv2 = torch.cat([z2xv2, e2_v], dim=1)
            else:
                zjointx1 = torch.cat([z1x1, e2], dim=1)
                zjointx2 = torch.cat([z2x2, e1], dim=1)
                zjointxv1 = torch.cat([z1xv1, e2_v], dim=1)
                zjointxv2 = torch.cat([z2xv2, e1_v], dim=1)

            if self.proj:
                zjointx1 = self.projection_x1(zjointx1)
                zjointx2 = self.projection_x2(zjointx2)
                zjointxv1 = self.projection_x1(zjointxv1)
                zjointxv2 = self.projection_x2(zjointxv2)

            zjointx1, zjointx2 = nn.functional.normalize(zjointx1, dim=-1), nn.functional.normalize(zjointx2, dim=-1)
            zjointxv1, zjointxv2 = nn.functional.normalize(zjointxv1, dim=-1), nn.functional.normalize(zjointxv2, dim=-1)
            concat_embed_x1 = torch.cat([zjointx1.unsqueeze(dim=1), zjointxv1.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([zjointx2.unsqueeze(dim=1), zjointxv2.unsqueeze(dim=1)], dim=1)
        else:
            z1x1_norm, z2x2_norm = nn.functional.normalize(z1x1, dim=-1), nn.functional.normalize(z2x2, dim=-1)
            z1xv1_norm, z2xv2_norm = nn.functional.normalize(z1xv1, dim=-1), nn.functional.normalize(z2xv2, dim=-1)
            concat_embed_x1 = torch.cat([z1x1_norm.unsqueeze(dim=1), z1xv1_norm.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([z2x2_norm.unsqueeze(dim=1), z2xv2_norm.unsqueeze(dim=1)], dim=1)

        specific_loss_x1, loss_x1, loss_y1 = self.critic(concat_embed_x1)
        specific_loss_x2, loss_x2, loss_y2 = self.critic(concat_embed_x2)

        loss_specific = specific_loss_x1 + specific_loss_x2

        if self.lmd_end_value > 0:
            lmd = self.lmd_scheduler(self.iterations)
        else:
            lmd = self.lmd_start_value

        loss_ortho = 0.5 * (ortho_loss(z1x1, e1, norm=self.ortho_norm) + ortho_loss(z2x2, e2, norm=self.ortho_norm)) + \
                    0.5 * (ortho_loss(z1xv1, e1_v, norm=self.ortho_norm) + ortho_loss(z2xv2, e2_v, norm=self.ortho_norm))
        
        loss = 2 * loss_shared/(1+self.a) + self.a * loss_specific/(1+self.a) + lmd * loss_ortho

        return loss, {'loss': loss.item(), 'shared': loss_shared.item(), 'clip': joint_loss.item(), 'loss_x': loss_x.item(), 'loss_y': loss_y.item(),
                       'specific': loss_specific.item(), 'ortho': loss_ortho.item(), 'lmd': lmd}#, 'beta': beta,
    
    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        zsx1 = self.encoder_x1s(x1)
        zsx2 = self.encoder_x2s(x2)
        if self.condzs:
            z1x1 = self.encoder_x1(torch.cat([x1, zsx1], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, zsx2], dim=1))
        else:
            z1x1 = self.encoder_x1(x1)
            z2x2 = self.encoder_x2(x2)
        return zsx1, zsx2, z1x1, z2x2
    
    def train_model(self, train_loader, train_dataset, test_dataset, optimizer, num_epoch=50, noise_scale=0.01, drop_scale=10):
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
        for _iter in range(num_epoch):
            logs = {}
            logs.update({'Epoch': _iter})
            loss_meter = utils.AverageMeter('loss')
            shared_meter = utils.AverageMeter('shared')
            clip_meter = utils.AverageMeter('clip')
            loss_x_meter = utils.AverageMeter('loss_x')
            loss_y_meter = utils.AverageMeter('loss_y')
            specific_meter = utils.AverageMeter('specific')
            ortho_meter = utils.AverageMeter('ortho')
            lmd_meter = utils.AverageMeter('lmd')

            for i_batch, data_batch in enumerate(train_loader):
                self.train()
                x1_raw = data_batch[0].float().cuda()
                x2_raw = data_batch[1].float().cuda()

                x1 = augment_data(x1_raw, noise_scale, drop_scale)
                x2 = augment_data(x2_raw, noise_scale, drop_scale)
                v1 = augment_data(x1_raw, noise_scale, drop_scale)
                v2 = augment_data(x2_raw, noise_scale, drop_scale)

                loss, train_logs = model(x1, x2, v1, v2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_meter.update(train_logs['loss'])
                shared_meter.update(train_logs['shared'])
                clip_meter.update(train_logs['clip'])
                loss_x_meter.update(train_logs['loss_x'])
                loss_y_meter.update(train_logs['loss_y'])
                specific_meter.update(train_logs['specific'])
                ortho_meter.update(train_logs['ortho'])
                lmd_meter.update(train_logs['lmd'])
                
                def getemb(dataset):
                    zsx, zsy, z1x, z2y = self.get_embedding(torch.stack(dataset[:][:-3]).cuda())
                    zs = torch.cat([zsx, zsy], dim=1).cpu().detach().numpy()
                    z1x = z1x.cpu().detach().numpy()
                    z2y = z2y.cpu().detach().numpy()
                    return zs, z1x, z2y
                
                def linearprobe(train_dataset, test_dataset):
                    train_zs, train_z1x, train_z2y = getemb(train_dataset)
                    test_zs, test_z1x, test_z2y = getemb(test_dataset)
                    score_s = linearprobe_acc(train_zs, test_zs, train_dataset, test_dataset)
                    score_spe1 = linearprobe_acc(train_z1x, test_z1x, train_dataset, test_dataset)
                    score_spe2 = linearprobe_acc(train_z2y, test_z2y, train_dataset, test_dataset)
                    return (score_s, score_spe1, score_spe2)
                
                def linearprobe_acc(train_z, test_z, train_dataset, test_dataset):
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-3])
                    score1 = clf.score(test_z, test_dataset[:][-3])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-2])
                    score2 = clf.score(test_z, test_dataset[:][-2])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-1])
                    score3 = clf.score(test_z, test_dataset[:][-1])
                    return (score1, score2, score3)

                if _iter == 0 and i_batch == 0:
                    logs.update({'loss': 0, 'shared': 0, 'clip': 0, 'loss_x': 0, 'loss_y': 0, 'specific': 0, 'ortho': 0, 'lmd': 0})
                    logs.update({'Acc_s 1': 0, 'Acc_s 2': 0, 'Acc_s 3': 0,
                                 'Acc_spe1 1': 0, 'Acc_spe1 2': 0, 'Acc_spe1 3': 0,
                                 'Acc_spe2 1': 0, 'Acc_spe2 2': 0, 'Acc_spe2 3': 0})
                    utils.print_row([i for i in logs.keys()], colwidth=12)
                
                if i_batch == len(train_loader)-1:
                    logs.update({'loss': loss_meter.avg, 'shared': shared_meter.avg, 'clip': clip_meter.avg,
                                    'loss_x': loss_x_meter.avg, 'loss_y': loss_y_meter.avg, 'specific': specific_meter.avg, 'ortho': ortho_meter.avg,
                                    'lmd': lmd_meter.avg})
                    
            if _iter == num_epoch-1:
                test_acc = linearprobe(train_dataset, test_dataset)
                logs.update({'Acc_s 1': test_acc[0][0], 'Acc_s 2': test_acc[0][1], 'Acc_s 3': test_acc[0][2],
                                'Acc_spe1 1': test_acc[1][0], 'Acc_spe1 2': test_acc[1][1], 'Acc_spe1 3': test_acc[1][2],
                                'Acc_spe2 1': test_acc[2][0], 'Acc_spe2 2': test_acc[2][1], 'Acc_spe2 3': test_acc[2][2]})                
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)        
            else:
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)  

            lr_scheduler.step()
        return logs
    

class Focal(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, layers=2, activation='relu', initialization='xavier',  
                a=1, lmd=1e-3):
        super().__init__()
        self.encoder_x1s = mlp(x1_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        self.encoder_x2s = mlp(x2_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        self.encoder_x1 = mlp(x1_dim+embed_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        self.encoder_x2 = mlp(x2_dim+embed_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        
        self.a = a
        self.lmd = lmd
        self.embed_dim = embed_dim

        self.critic_c = SupConLoss()
        self.critic_s = SupConLoss()

    def forward(self, x1, x2, v1, v2):
        e1_c = self.encoder_x1s(x1)
        e2_c = self.encoder_x2s(x2)
        e1_c_v = self.encoder_x1s(v1)
        e2_c_v = self.encoder_x2s(v2)
        e1_s = self.encoder_x1(torch.cat([x1, e1_c], dim=1))
        e2_s = self.encoder_x2(torch.cat([x2, e2_c], dim=1))
        e1_s_v = self.encoder_x1(torch.cat([v1, e1_c_v], dim=1))
        e2_s_v = self.encoder_x2(torch.cat([v2, e2_c_v], dim=1))

        e1_c, e2_c = F.normalize(e1_c, dim=-1), F.normalize(e2_c, dim=-1)
        e1_c_v, e2_c_v = F.normalize(e1_c_v, dim=-1), F.normalize(e2_c_v, dim=-1)
        e1_s, e2_s = F.normalize(e1_s, dim=-1), F.normalize(e2_s, dim=-1)
        e1_s_v, e2_s_v = F.normalize(e1_s_v, dim=-1), F.normalize(e2_s_v, dim=-1)

        concat_embed_c = torch.cat([e1_c.unsqueeze(1), e2_c.unsqueeze(1)], dim=1)
        concat_embed_c_v = torch.cat([e1_c_v.unsqueeze(1), e2_c_v.unsqueeze(1)], dim=1)
        joint_loss_c, _, _ = self.critic_c(concat_embed_c)
        joint_loss_c_v, _, _ = self.critic_c(concat_embed_c_v)
        joint_loss = joint_loss_c + joint_loss_c_v

        concat_embed_s1 = torch.cat([e1_s.unsqueeze(1), e1_s_v.unsqueeze(1)], dim=1)
        concat_embed_s2 = torch.cat([e2_s.unsqueeze(1), e2_s_v.unsqueeze(1)], dim=1)
        specific_loss_s1, _, _ = self.critic_s(concat_embed_s1)
        specific_loss_s2, _, _ = self.critic_s(concat_embed_s2)
        loss_specific = specific_loss_s1 + specific_loss_s2

        loss_ortho = ortho_loss_focal(e1_s, e1_c) + ortho_loss_focal(e2_s, e2_c) + \
                        ortho_loss_focal(e1_s_v, e1_c_v) + ortho_loss_focal(e2_s_v, e2_c_v) + \
                        ortho_loss_focal(e1_s, e2_s) + ortho_loss_focal(e1_s_v, e2_s_v)
        
        loss = joint_loss + self.a * loss_specific + self.lmd * loss_ortho
        return loss, {'loss': loss.item(), 'loss_shared': joint_loss.item(), 'loss_specific': loss_specific.item(), 'loss_ortho': loss_ortho.item()}
    
    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        e1_c = self.encoder_x1s(x1)
        e2_c = self.encoder_x2s(x2)
        e1_s = self.encoder_x1(torch.cat([x1, e1_c], dim=1))
        e2_s = self.encoder_x2(torch.cat([x2, e2_c], dim=1))
        return e1_c, e2_c, e1_s, e2_s
    
    def train_model_focal(self, train_loader, train_dataset, test_dataset, optimizer, num_epoch=50, noise_scale=0.01, drop_scale=10):
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
        for _iter in range(num_epoch):
            logs = {}
            logs.update({'Epoch': _iter})
            loss_meter = utils.AverageMeter('loss')
            shared_meter = utils.AverageMeter('shared')
            specific_meter = utils.AverageMeter('specific')
            ortho_meter = utils.AverageMeter('ortho')
            lmd_meter = utils.AverageMeter('lmd')

            for i_batch, data_batch in enumerate(train_loader):
                self.train()
                x1 = data_batch[0].float().cuda()
                x2 = data_batch[1].float().cuda()
                x1 = augment_data(x1, noise_scale, drop_scale)
                x2 = augment_data(x2, noise_scale, drop_scale)
                v1 = augment_data(x1, noise_scale, drop_scale)
                v2 = augment_data(x2, noise_scale, drop_scale)
                loss, train_logs = self(x1, x2, v1, v2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_meter.update(train_logs['loss'])
                shared_meter.update(train_logs['loss_shared'])
                specific_meter.update(train_logs['loss_specific'])
                ortho_meter.update(train_logs['loss_ortho'])
                
                def getemb(dataset):
                    zsx, zsy, z1x, z2y = self.get_embedding(torch.stack(dataset[:][:-3]).cuda())
                    zs = torch.cat([zsx, zsy], dim=1).cpu().detach().numpy()
                    z1x = z1x.cpu().detach().numpy()
                    z2y = z2y.cpu().detach().numpy()
                    return zs, z1x, z2y
                
                def linearprobe(train_dataset, test_dataset):
                    train_zs, train_z1x, train_z2y = getemb(train_dataset)
                    test_zs, test_z1x, test_z2y = getemb(test_dataset)
                    score_s = linearprobe_acc(train_zs, test_zs, train_dataset, test_dataset)
                    score_spe1 = linearprobe_acc(train_z1x, test_z1x, train_dataset, test_dataset)
                    score_spe2 = linearprobe_acc(train_z2y, test_z2y, train_dataset, test_dataset)
                    return (score_s, score_spe1, score_spe2)
                
                def linearprobe_acc(train_z, test_z, train_dataset, test_dataset):
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-3])
                    score1 = clf.score(test_z, test_dataset[:][-3])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-2])
                    score2 = clf.score(test_z, test_dataset[:][-2])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-1])
                    score3 = clf.score(test_z, test_dataset[:][-1])
                    return (score1, score2, score3)

                if _iter == 0 and i_batch == 0:
                    logs.update({'loss': 0, 'shared': 0, 'specific': 0, 'ortho': 0})
                    logs.update({'Acc_s 1': 0, 'Acc_s 2': 0, 'Acc_s 3': 0,
                                 'Acc_spe1 1': 0, 'Acc_spe1 2': 0, 'Acc_spe1 3': 0,
                                 'Acc_spe2 1': 0, 'Acc_spe2 2': 0, 'Acc_spe2 3': 0})
                    utils.print_row([i for i in logs.keys()], colwidth=12)
                
                if i_batch == len(train_loader)-1:
                    logs.update({'loss': loss_meter.avg, 'shared': shared_meter.avg, 'specific': specific_meter.avg, 'ortho': ortho_meter.avg})
                    
            if _iter%5 ==0 or _iter == num_epoch-1:
                test_acc = linearprobe(train_dataset, test_dataset)
                logs.update({'Acc_s 1': test_acc[0][0], 'Acc_s 2': test_acc[0][1], 'Acc_s 3': test_acc[0][2],
                                'Acc_spe1 1': test_acc[1][0], 'Acc_spe1 2': test_acc[1][1], 'Acc_spe1 3': test_acc[1][2],
                                'Acc_spe2 1': test_acc[2][0], 'Acc_spe2 2': test_acc[2][1], 'Acc_spe2 3': test_acc[2][2]})                
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)        
            else:
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)  

            lr_scheduler.step()
        return logs
    

class DMVAE(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, layers=2, activation='relu', initialization='xavier', a=1):
        super().__init__()
        self.encoder_x1 = mlp(x1_dim, hidden_dim, 4*embed_dim, layers, activation, initialization = initialization)
        self.encoder_x2 = mlp(x2_dim, hidden_dim, 4*embed_dim, layers, activation, initialization = initialization)

        self.decoder_x1 = mlp(2*embed_dim, hidden_dim, x1_dim, layers, activation, initialization = initialization)
        self.decoder_x2 = mlp(2*embed_dim, hidden_dim, x2_dim, layers, activation, initialization = initialization)
        
        self.a = a
        self.embed_dim = embed_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def product_of_expert(self, mu1, logvar1, mu2, logvar2):
        logvar = - (1/logvar1.exp() + 1/logvar2.exp()).log()
        mu = logvar.exp() * (mu1 / logvar1.exp() + mu2 / logvar2.exp())
        return mu, logvar
    
    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x1, x2):
        emb1 = self.encoder_x1(x1)
        emb2 = self.encoder_x2(x2)
        mu_s1, logvar_s1, mu_1, logvar_1 = emb1.chunk(4, dim=1)
        mu_s2, logvar_s2, mu_2, logvar_2 = emb2.chunk(4, dim=1)
        z1 = self.reparameterize(mu_1, logvar_1)
        z2 = self.reparameterize(mu_2, logvar_2)
        z1_s = self.reparameterize(mu_s1, logvar_s1)
        z2_s = self.reparameterize(mu_s2, logvar_s2)
        mu_s, logvar_s = self.product_of_expert(mu_s1, logvar_s1, mu_s2, logvar_s2)
        z_s = self.reparameterize(mu_s, logvar_s)
        x1_recon = self.decoder_x1(torch.cat([z1, z_s], dim=1))
        x2_recon = self.decoder_x2(torch.cat([z2, z_s], dim=1))
        loss_recon = F.mse_loss(x1_recon, x1) + F.mse_loss(x2_recon, x2)
        loss_kl = self.kl_divergence(mu_1, logvar_1) + self.kl_divergence(mu_2, logvar_2) + 2 * self.kl_divergence(mu_s, logvar_s)

        x1_recon_cross = self.decoder_x1(torch.cat([z1, z2_s], dim=1))
        x2_recon_cross = self.decoder_x2(torch.cat([z2, z1_s], dim=1))
        loss_recon_cross = F.mse_loss(x1_recon_cross, x1) + F.mse_loss(x2_recon_cross, x2)
        loss_kl_cross = self.kl_divergence(mu_1, logvar_2) + self.kl_divergence(mu_2, logvar_1) + self.kl_divergence(mu_s1, logvar_s1) + self.kl_divergence(mu_s2, logvar_s2)

        loss = loss_recon + self.a * loss_kl + loss_recon_cross + self.a * loss_kl_cross
        return loss, {'loss': loss.item(), 'loss_recon': loss_recon.item(), 'loss_kl': loss_kl.item(), 'loss_recon_cross': loss_recon_cross.item(), 'loss_kl_cross': loss_kl_cross.item(), 'a': self.a}

    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        emb1 = self.encoder_x1(x1)
        emb2 = self.encoder_x2(x2)
        mu_s1, logvar_s1, mu_1, logvar_1 = emb1.chunk(4, dim=1)
        mu_s2, logvar_s2, mu_2, logvar_2 = emb2.chunk(4, dim=1)
        return mu_s1, mu_s2, mu_1, mu_2
    
    def train_model_dmvae(self, train_loader, train_dataset, test_dataset, optimizer, num_epoch=50, noise_scale=0.01, drop_scale=10):
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
        for _iter in range(num_epoch):
            logs = {}
            logs.update({'Epoch': _iter})
            loss_meter = utils.AverageMeter('loss')
            recon_meter = utils.AverageMeter('recon')
            kl_meter = utils.AverageMeter('kl')
            recon_cross_meter = utils.AverageMeter('recon_cross')
            kl_cross_meter = utils.AverageMeter('kl_cross')
            a_meter = utils.AverageMeter('a')

            for i_batch, data_batch in enumerate(train_loader):
                self.train()
                x1 = data_batch[0].float().cuda()
                x2 = data_batch[1].float().cuda()
                x1 = augment_data(x1, noise_scale, drop_scale)
                x2 = augment_data(x2, noise_scale, drop_scale)
                loss, train_logs = self(x1, x2) #, v1, v2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_meter.update(train_logs['loss'])
                recon_meter.update(train_logs['loss_recon'])
                kl_meter.update(train_logs['loss_kl'])
                recon_cross_meter.update(train_logs['loss_recon_cross'])
                kl_cross_meter.update(train_logs['loss_kl_cross'])
                a_meter.update(train_logs['a'])
                
                def getemb(dataset):
                    zsx, zsy, z1x, z2y = self.get_embedding(torch.stack(dataset[:][:-3]).cuda())
                    zs = torch.cat([zsx, zsy], dim=1).cpu().detach().numpy()
                    z1x = z1x.cpu().detach().numpy()
                    z2y = z2y.cpu().detach().numpy()
                    return zs, z1x, z2y
                
                def linearprobe(train_dataset, test_dataset):
                    train_zs, train_z1x, train_z2y = getemb(train_dataset)
                    test_zs, test_z1x, test_z2y = getemb(test_dataset)
                    score_s = linearprobe_acc(train_zs, test_zs, train_dataset, test_dataset)
                    score_spe1 = linearprobe_acc(train_z1x, test_z1x, train_dataset, test_dataset)
                    score_spe2 = linearprobe_acc(train_z2y, test_z2y, train_dataset, test_dataset)
                    return (score_s, score_spe1, score_spe2)
                
                def linearprobe_acc(train_z, test_z, train_dataset, test_dataset):
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-3])
                    score1 = clf.score(test_z, test_dataset[:][-3])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-2])
                    score2 = clf.score(test_z, test_dataset[:][-2])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-1])
                    score3 = clf.score(test_z, test_dataset[:][-1])
                    return (score1, score2, score3)

                if _iter == 0 and i_batch == 0:
                    logs.update({'loss': 0, 'recon': 0, 'kl': 0, 'recon_cross': 0, 'kl_cross': 0, 'a': 0})
                    logs.update({'Acc_s 1': 0, 'Acc_s 2': 0, 'Acc_s 3': 0,
                                 'Acc_spe1 1': 0, 'Acc_spe1 2': 0, 'Acc_spe1 3': 0,
                                 'Acc_spe2 1': 0, 'Acc_spe2 2': 0, 'Acc_spe2 3': 0})
                    utils.print_row([i for i in logs.keys()], colwidth=12)
                
                if i_batch == len(train_loader)-1:
                    logs.update({'loss': loss_meter.avg, 'recon': recon_meter.avg, 'kl': kl_meter.avg, 'recon_cross': recon_cross_meter.avg, 'kl_cross': kl_cross_meter.avg, 'a': a_meter.avg})
                    
            if _iter%5 ==0 or _iter == num_epoch-1:
                test_acc = linearprobe(train_dataset, test_dataset)
                logs.update({'Acc_s 1': test_acc[0][0], 'Acc_s 2': test_acc[0][1], 'Acc_s 3': test_acc[0][2],
                                'Acc_spe1 1': test_acc[1][0], 'Acc_spe1 2': test_acc[1][1], 'Acc_spe1 3': test_acc[1][2],
                                'Acc_spe2 1': test_acc[2][0], 'Acc_spe2 2': test_acc[2][1], 'Acc_spe2 3': test_acc[2][2]})                
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)        
            else:
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)  

            lr_scheduler.step()
        return logs
