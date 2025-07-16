import random
from typing import Union
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from ml.models.autoencoder import VariationalAutoEncoder, AutoEncoder
from ml.models.mlp import MLP
from ml.utils.data_utils import TorchDataset

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

thefts = pd.read_csv("dataset/final/valid_train_thefts.csv")
non_thefts = pd.read_csv("dataset/final/valid_train_non_theft_suc_theft_acct.csv")

df = pd.concat([thefts, non_thefts])
df.drop(columns=['acct', 'successor', 'voltage'], inplace=True)
df.fillna(-1, inplace=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

scaler = MinMaxScaler()

# define X and y
X = df.drop('target', axis=1)
y = df['target']

acc_scores = []
prec_scores = []
rec_scores = []
f1_scores = []
brier_scores, auc_roc_scores, pr_auc_scores = [], [], []
lift_scores, demotion_scores, weighted_scores = [], [], []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = TorchDataset(X=torch.tensor(X_train).float(),
                                 y=torch.tensor(y_train.to_numpy()).float())  # reconstruction
    test_dataset = TorchDataset(X=torch.tensor(X_test).float(), y=torch.tensor(y_test.to_numpy()).float())

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    encoder = MLP(in_size=X_train.shape[1], layer_units=[32, 64], out_size=128, vae=False)
    decoder = MLP(in_size=128, layer_units=[64, 32], out_size=X_train.shape[1])

    model = AutoEncoder(encoder, decoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = dict(model=model, train_loader=train_loader, test_loader=test_loader,
                epochs=1000, optimizer=optimizer, criterion=criterion,
                reconstruction=True, vae=False, device=device, verbose=True, return_best=True,
                plot_history=False,
                num_test_samples=100
                )
    args = SimpleNamespace(**args)
    model: Union[MLP, VariationalAutoEncoder, AutoEncoder] = model.fit(args)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_latent, test_latent = [], []
    for tmp_x, _ in train_loader:
        tmp_x = tmp_x.to(device)
        z = model.encode(tmp_x)
        train_latent.append(z)
    for tmp_x, _ in test_loader:
        tmp_x = tmp_x.to(device)
        z = model.encode(tmp_x)
        test_latent.append(z)

    train_latent = torch.cat(train_latent)
    test_latent = torch.cat(test_latent)

    train_dataset = TorchDataset(X=train_latent, y=torch.tensor(y_train.to_numpy()).float())
    test_dataset = TorchDataset(X=test_latent, y=torch.tensor(y_test.to_numpy()).float())

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    classifier = MLP(in_size=train_latent.shape[1], layer_units=[128, 64, 8], out_size=1)
    print(classifier)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = dict(model=classifier, train_loader=train_loader, test_loader=test_loader,
                epochs=1000, optimizer=optimizer, criterion=criterion,
                reconstruction=False, vae=False, device=device, verbose=True, return_best=True, plot_history=False,
                num_test_samples=100)
    args = SimpleNamespace(**args)
    classifier: Union[MLP, VariationalAutoEncoder] = classifier.fit(args)

    args = dict(
        model=classifier,
        data_loader=test_loader, criterion=criterion,
        reconstruction=False, vae=False, device=device)
    args = SimpleNamespace(**args)

    _, acc, precision, recall, f1, brier, roc_auc, pr_auc, lift, demotion, weighted_score = classifier.predict(
        args
    )

    acc_scores.append(acc)
    prec_scores.append(precision)
    rec_scores.append(recall)
    f1_scores.append(f1)

    brier_scores.append(brier)
    auc_roc_scores.append(roc_auc)
    pr_auc_scores.append(pr_auc)

    lift_scores.append(lift)
    demotion_scores.append(demotion)
    weighted_scores.append(weighted_score)

    print(
        "Fold {} - Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Brier: {:.4f}, AUC-ROC: {:.4f}, "
        "PR-AUC: {:.4f}, Lift: {:.4f}, Demotion: {:.4f}, Weighted: {:.4f}"
        .format(len(acc_scores), acc_scores[-1], prec_scores[-1], rec_scores[-1], f1_scores[-1],
                brier_scores[-1], auc_roc_scores[-1], pr_auc_scores[-1],
                lift_scores[-1], demotion_scores[-1], weighted_scores[-1]))
    input("??")

print(
    "Average - Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Brier: {:.4f}, AUC-ROC: {:.4f}, "
    "PR-AUC: {:.4f}, Lift: {:.4f}, Demotion: {:.4f}, Weighted: {:.4f}\n"
    .format(sum(acc_scores) / len(acc_scores), sum(prec_scores) / len(prec_scores),
            sum(rec_scores) / len(rec_scores), sum(f1_scores) / len(f1_scores),
            sum(brier_scores) / len(brier_scores), sum(auc_roc_scores) / len(auc_roc_scores),
            sum(pr_auc_scores) / len(pr_auc_scores),
            sum(lift_scores) / len(lift_scores), sum(demotion_scores) / len(demotion_scores),
            sum(weighted_scores) / len(weighted_scores)))
