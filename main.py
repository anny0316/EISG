from datetime import datetime
import logging
import random
from sklearn.metrics import roc_auc_score, roc_auc_score
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, DenseDataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm

from informal.utils import plot_curves_new, visualize_embeddings, setup_logger
from models.GCN import GCN
from models.GIN import GIN
from models.GAT import GAT
from models.GIB import GIB, JointGenerator, Discriminator
from models.mygin import MYGIN
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from sklearn.manifold import TSNE
import numpy as np
from datasets.get_dataset import get_dataset
from train_eval import train_ib, train_ast, train_ib_with_env, test, train
import argparse
from mmcv import Config
from datasets.drugood_dataset import DrugOOD
from drugood.datasets import build_dataset, build_dataloader
from drugood.models import build_backbone
import os
from torch_geometric.loader import DataLoader
from models.env_inference import GraphFeatureExtractor, Predictor, ConditionalGnn, DomainClassifier


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str,
        help='the config for building dataset',
        # update
        default='drugood_lbap_core_ec50_assay'
    )
    parser.add_argument(
        '--seed', default=42, type=int,
        help='the seed of training'
    )
    parser.add_argument(
        '--dist', default='uniform', type=str,
        help='the prior distribution of ELBO'
    )
    parser.add_argument('--num_domain', type=int, default=8)
    # parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--root', type=str,default='data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--epochs_env', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dataset_name', type=str, default='DrugOOD')
    parser.add_argument('--train_model', type=str, default='ib_env')
    parser.add_argument('--kld_weight', type=float, default=0.1)
    parser.add_argument('--env_weight', type=float, default=2)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    return args

def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)



if __name__ == "__main__":
    args = init_args()
    device = args.device
    random_seed = random.randint(0, 100000)
    print(f"Random seed: {random_seed}")
    seed_everything(random_seed)
    if args.dataset_name == 'DrugOOD':
        config_path = os.path.join("configs", args.dataset + ".py")
        cfg = Config.fromfile(config_path)
        root = os.path.join(args.root, "DrugOOD")

        train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=args.dataset, mode="train")
        val_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_val), name=args.dataset, mode="ood_val")
        test_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_test), name=args.dataset, mode="ood_test")
        print(f"Train Dataset size: {len(train_dataset)}")
        print(f"Validation Dataset size: {len(val_dataset)}")
        print(f"Test Dataset size: {len(test_dataset)}")
        cfg.data.ood_val.test_mode = True
        cfg.data.ood_test.test_mode = True

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        print(train_dataset.num_classes)
        num_features = train_dataset.num_features
        num_classes = train_dataset.num_classes
    else:
        dataset = get_dataset(args.dataset_name, sparse=True)

        train_size = int(0.8 * len(dataset))
        valid_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - valid_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
        print(f"Train Dataset size: {len(train_dataset)}")
        print(f"Validation Dataset size: {len(val_dataset)}")
        print(f"Test Dataset size: {len(test_dataset)}")
        if 'adj' in train_dataset[0]:
            train_loader = DenseDataLoader(train_dataset, args.batch_size, shuffle=True)
            val_loader = DenseDataLoader(val_dataset, args.batch_size, shuffle=False)
            test_loader = DenseDataLoader(test_dataset, args.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
            valid_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        num_features = dataset.num_features
        num_classes = dataset.num_classes

    if args.train_model == 'ib':
        model = MYGIN(num_features=num_features, num_classes=num_classes, hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers).to(device)
        model.to(device).reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.train_model =='env':
        backend = GraphFeatureExtractor(num_features=num_features, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        domain_classifier = DomainClassifier(backend_dim=args.hidden_dim, backend=backend, num_domain=args.num_domain, num_task=1)
        conditional_gnn = ConditionalGnn(emb_dim=args.hidden_dim, backend_dim=args.hidden_dim, backend=backend, num_domain=args.num_domain, num_class=num_classes)

        domain_classifier.to(device).reset_parameters()
        optimizer_dom = torch.optim.Adam(domain_classifier.parameters(), lr=args.lr)

        conditional_gnn.to(device).reset_parameters()
        optimizer_con = torch.optim.Adam(conditional_gnn.parameters(), lr=args.lr)
    elif args.train_model =='ib_env':
        model = MYGIN(num_features=num_features,  hidden_dim=args.hidden_dim,
                      num_layers=args.num_layers,num_classes=num_classes,jk_mode='max',device=device).to(device)
        model.to(device).reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        backend = GraphFeatureExtractor(num_features=num_features, hidden_dim=args.hidden_dim, num_layers=num_classes)
        domain_classifier = DomainClassifier(backend_dim=args.hidden_dim, backend=backend, num_domain=args.num_domain, num_task=1)
        conditional_gnn = ConditionalGnn(emb_dim=args.hidden_dim, backend_dim=args.hidden_dim, backend=backend, num_domain=args.num_domain, num_class=num_classes)
        domain_classifier.to(device).reset_parameters()
        optimizer_dom = torch.optim.Adam(domain_classifier.parameters(), lr=args.lr)
        conditional_gnn.to(device).reset_parameters()
        optimizer_con = torch.optim.Adam(conditional_gnn.parameters(), lr=args.lr)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GIN(num_features=39, num_classes=2, hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_losses = []
    val_losses = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []
    test_aucs = []
    best_auc = 0
    best_epoch = 0
    best_epoch_env = 0
    best_acc = 0.0  #
    best_acc_epoch = 0  #
    best_elbo = float('inf')
    best_conditional_gnn_state_dict = None
    best_domain_classifier_state_dict = None
    best_val_acc = 0

    # 训练环境推理模型
    if args.train_model == 'ib_env':
        print('训练环境推理模型')
        for epoch in range(1, args.epochs_env + 1):
            eq, elbo, conditional_gnn, domain_classifier = train_ast(conditional_gnn, domain_classifier,
                                                                     optimizer_con,
                                                                     optimizer_dom,
                                                                     train_loader, device, args)
            print(f'Epoch {epoch}: eq: {eq:.4f}, elbo: {elbo:.4f}')
            if elbo < best_elbo:
                best_elbo = elbo
                best_conditional_gnn_state_dict = conditional_gnn.state_dict()
                best_domain_classifier_state_dict = domain_classifier.state_dict()
                best_epoch_env = epoch

    print(f"加载了第 {best_epoch_env} 轮的环境推理模型, val_acc: {best_val_acc}")
    for epoch in range(1, args.epochs + 1):
        if args.train_model == 'ib_env':
            domain_classifier.load_state_dict(best_domain_classifier_state_dict)
            train_loss = train_ib_with_env(domain_classifier, model, optimizer,
                                           train_loader, device, args)
            train_losses.append(train_loss)
            val_acc, val_loss, _ = test(model, valid_loader, device)
            test_acc, test_loss, test_auc = test(model, test_loader, device)
            val_accuracies.append(val_acc)
            test_aucs.append(test_auc)
            test_accuracies.append(test_acc)
            if test_auc > best_auc:
                best_auc = test_auc
                best_epoch = epoch
            if test_acc > best_acc:
                best_acc = test_acc
                best_acc_epoch = epoch

            print(
                f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}')
        elif args.train_model == 'ib':
            # 主模型训练 (IB 模式)
            train_loss = train_ib(model,  optimizer,train_loader, args.kld_weight,device)
            train_losses.append(train_loss)
            val_acc, val_loss, _ = test(model, valid_loader, device)
            test_acc, test_loss, test_auc = test(model, test_loader, device)
            val_accuracies.append(val_acc)
            test_aucs.append(test_auc)
            test_accuracies.append(test_acc)
            if test_auc > best_auc:
                best_auc = test_auc
                best_epoch = epoch
            if test_acc > best_acc:
                best_acc = test_acc
                best_acc_epoch = epoch
            print(
                f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}')
        elif args.train_model == 'env':
            eq, elbo, conditional_gnn, domain_classifier = train_ast(conditional_gnn, domain_classifier,
                                                                     optimizer_con, optimizer_dom,
                                                                     train_loader, device, args)

        else:
            train_loss = train(model, train_loader, optimizer, device)
            val_acc, val_loss, _ = test(model, valid_loader, device)
            test_acc, test_loss, test_auc = test(model, test_loader, device)
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            test_accuracies.append(test_acc)
            test_aucs.append(test_auc)

    print(
        f'Best AUC: {best_auc:.4f} achieved at epoch {best_epoch} Best ACC: {best_acc:.4f} achieved at epoch {best_acc_epoch}')
    plot_curves_new(train_losses, val_accuracies, test_accuracies, test_aucs, args.epochs)


