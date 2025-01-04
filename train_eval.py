import time
from copy import deepcopy
from itertools import chain
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.distributions.normal import Normal

from informal.losses import MeanLoss, KLDist
from informal.utils import set_masks, num_graphs, clear_masks, get_prior


def train_ib_with_env(domain_classifier, model, optimizer_model,
                  train_loader, device, args):


    CLSLoss = torch.nn.CrossEntropyLoss()
    mean_loss = MeanLoss(CLSLoss)
    domain_classifier = domain_classifier.eval()
    total_loss = 0
    for data in tqdm(train_loader):
        data = data.to(device)

        with torch.no_grad():
            p_e = domain_classifier(data)
            group = torch.argmax(p_e, dim=-1)

        sub_pre, kld_loss, pre = model(data)

        loss_local = F.nll_loss(sub_pre, data.y.view(-1))
        mean_term,_,_ = mean_loss(pre, data.y.long(), group)

        optimizer_model.zero_grad()
        loss = loss_local +  args.kld_weight*kld_loss+ args.env_weight*mean_term
        total_loss += loss.item() * num_graphs(data)
        loss.backward()
        optimizer_model.step()

    mean_loss = total_loss / len(train_loader.dataset)
    return mean_loss

def train_ast(conditional_gnn, domain_classifier, optimizer_con, optimizer_dom, train_loader, device, args):
    conditional_gnn.train()
    domain_classifier.train()
    Eqs, ELs = [], []
    KLDs = []
    prior = get_prior(args.num_domain, args.dist).to(device)

    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer_con.zero_grad()
        optimizer_dom.zero_grad()

        q_e = torch.softmax(domain_classifier(data), dim=-1)
        losses, batch_size = [], len(data.y)

        for dom in range(args.num_domain):
            domain_info = torch.ones(batch_size).long().to(device) * dom
            p_ye = conditional_gnn(data, domain_info)
            loss = F.cross_entropy(p_ye, data.y.long(), reduction='none')
            losses.append(loss)

        losses = torch.stack(losses, dim=1)
        Eq = torch.mean(torch.sum(q_e * losses, dim=-1))
        kl_loss = KLDist(q_e, prior)
        ELBO = Eq + KLDist(q_e, prior)

        ELBO.backward()
        optimizer_con.step()
        optimizer_dom.step()

        Eqs.append(Eq.item())
        ELs.append(ELBO.item())
        KLDs.append(kl_loss.item())

    mean_Eq = np.mean(Eqs)
    mean_ELBO = np.mean(ELs)
    mean_KLD = np.mean(KLDs)
    print(mean_KLD)
    return mean_Eq, mean_ELBO, conditional_gnn, domain_classifier


def train_ib(model, optimizer_model,  loader, kld_weight, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader):

        data = data.to(device)

        sub_pre, kld_loss, x = model(data)

        loss_local = F.nll_loss(sub_pre, data.y.view(-1))


        optimizer_model.zero_grad()

        loss = loss_local + kld_weight * kld_loss
        total_loss += loss.item() * num_graphs(data)
        loss.backward()
        optimizer_model.step()

    mean_loss = total_loss / len(loader.dataset)
    return mean_loss



def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        # check_data(data)
        optimizer.zero_grad()
        out,_,_ = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, loader, device):
    model.eval()
    correct = 0
    total_loss = 0
    all_log_probs = []
    all_labels = []
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            out, _ = model(data,mask=None,train=False)
            loss = F.nll_loss(out, data.y)
            total_loss += loss.item()
            pred = out.max(1)[1]
            correct += pred.eq(data.y).sum().item()
            all_log_probs.extend(out.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    all_log_probs = np.array(all_log_probs)
    all_labels = np.array(all_labels)

    all_probs = np.exp(all_log_probs)

    if len(np.unique(all_labels)) > 2:
        # 多分类情况
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
    else:
        # 二分类情况
        auc = roc_auc_score(all_labels, all_probs[:, 1])

    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader)

    return accuracy, avg_loss, auc