import copy
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch import optim

from ml.utils.metrics import get_classification_metrics, get_probability_measures, get_lift_demotion_scores


def fit(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int,
        optimizer: optim.Optimizer, criterion: torch.nn.Module, reconstruction: Optional[bool] = True,
        vae: Optional[bool] = False, verbose: Optional[bool] = True, return_best: Optional[bool] = True,
        plot_history: Optional[bool] = True, num_test_samples: Optional[int] = 100, device: Optional[str] = 'cuda'):
    """
    Trains and evaluates a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (torch.utils.data.DataLoader): The DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): The DataLoader for test data.
        epochs (int): The number of epochs to train the model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training and evaluation.
        reconstruction (bool, optional): Whether the model is trained for reconstruction or classification.
                                         Defaults to True.
        vae (bool, optional): Whether the model is a Variational Autoencoder. Defaults to False.
        verbose (bool, optional): Whether to print evaluation metrics during training. Defaults to True.
        return_best (bool, optional): Whether to return the best model based on the test loss or the weighted score
                                      defined by the organizers. Defaults to True.
        plot_history (bool, optional): Whether to plot the training and test loss history. Defaults to True.
        num_test_samples (int, optional): The number of samples used to calculating lift and demotion scores.
                                          Defaults to 100. When the reconstruction flag is set to True, this parameter
                                          is not used.
        device: (str, optional): The device to use for training. Defaults to 'cuda'.

    Returns:
        torch.nn.Module: A trained model.
    """
    model.to(device)  # move the model to the specified device
    model.train()  # set the model to training mode
    best_loss, best_epoch = np.Inf, -1  # initialize vars to keep track of the best model
    best_model = copy.deepcopy(model)
    train_history, test_history = [], []  # initialize vars to keep track of the training and test loss

    test_weighted_score = None  # initialize the weighted score defined in the challenge.
    for epoch in range(1, epochs + 1):  # loop through epochs
        epoch_losses = []  # initialize a list to keep track of the loss per mini batch in the epoch
        for i, (x, y) in enumerate(train_loader):  # loop through batches
            # move the input features and targets to the specified device
            x = x.to(device)
            y = y.to(device)
            # zero out gradients
            optimizer.zero_grad()
            # make predictions
            out = model(x)

            # calculate loss
            loss = criterion(out, x) if reconstruction else criterion(out, y.reshape(-1, 1))

            # add the kl loss when using VAE
            if vae:
                kl_loss = model.kl_loss()
                loss += kl_loss
            # append batch loss
            epoch_losses.append(loss.item())
            # backpropagation
            loss.backward()
            # update weights
            optimizer.step()

        # get the average loss for the current epoch
        epoch_loss = sum(epoch_losses) / len(epoch_losses)

        # print evaluation metrics
        if not reconstruction and verbose:
            aux_train = predict(
                model, train_loader, criterion,
                reconstruction=reconstruction,
                num_samples=num_test_samples,
                device=device
            )
            _, train_acc, train_precision, train_recall, train_f1, train_brier, train_roc_auc, train_pr_auc, \
                train_lift, train_demotion, train_weighted_score = aux_train
            print(f"Epoch: {epoch}/{epochs}")
            print(
                f"[Train]\tLoss={epoch_loss}, Accuracy: {train_acc}, Precision: {train_precision}, "
                f"Recall: {train_recall}, F1: {train_f1}\n"
                f"\t\tBrier score loss: {train_brier}, ROC-AUC: {train_roc_auc}, PR-AUC: {train_pr_auc}\n"
                f"\t\tLift: {train_lift}, Demotion: {train_demotion}, Weighted Score: {train_weighted_score}")

            aux = predict(
                model, test_loader, criterion,
                reconstruction=reconstruction,
                num_samples=num_test_samples,
                device=device
            )
            test_loss, test_acc, test_precision, test_recall, test_f1, test_brier, test_roc_auc, test_pr_auc, \
                test_lift, test_demotion, test_weighted_score = aux

            try:
                test_loss = test_loss.item()
            except AttributeError:
                test_loss = test_loss

            print(
                f"[Test]\tLoss={test_loss}, Accuracy: {test_acc}, Precision: {test_precision}, "
                f"Recall: {test_recall}, F1: {test_f1}\n"
                f"\t\tBrier score loss: {test_brier}, ROC-AUC: {test_roc_auc}, PR-AUC: {test_pr_auc}\n"
                f"\t\tLift: {test_lift}, Demotion: {test_demotion}, Weighted Score: {test_weighted_score}\n")
        elif reconstruction and verbose:
            print(f"Epoch: {epoch}/{epochs}")
            print(f"[Train]\tLoss={epoch_loss}")
            if vae:
                print(f"KL Loss={kl_loss}")
            test_loss = predict(model, test_loader, criterion,
                                reconstruction=reconstruction, num_samples=num_test_samples, device=device)[0]
            print(f"[Test]\tLoss={test_loss}\n")
        else:
            try:
                test_loss = predict(model, test_loader, criterion,
                                    reconstruction=reconstruction, num_samples=num_test_samples, device=device)[0]
            except TypeError:
                test_loss = predict(model, test_loader, criterion,
                                    reconstruction=reconstruction, num_samples=num_test_samples, device=device)
            try:
                test_loss = test_loss.item()
            except AttributeError:
                test_loss = test_loss
            # print(f"[Train]\tLoss={epoch_loss}")
        if return_best:  # keep track of the best model
            if test_weighted_score is not None:
                if epoch == 1 and test_weighted_score is not None:
                    best_loss = -1.
                if best_loss <= test_weighted_score:
                    best_loss = test_weighted_score
                    best_epoch = epoch
                    best_model = copy.deepcopy(model)
            elif best_loss >= test_loss:
                best_loss = test_loss
                best_epoch = epoch
                best_model = copy.deepcopy(model)
        if plot_history:  # keep track the training and test history
            train_history.append(epoch_loss)
            test_history.append(test_loss)

    if not return_best:
        best_model = copy.deepcopy(model)
    else:
        print(f"Best loss ({best_loss:.4f}) found on epoch {best_epoch}")

    if plot_history:  # plot the training and test history
        plt.plot(train_history, label='Train')
        plt.plot(test_history, label='Test')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.show()
        plt.close()

    return best_model


def predict(model: torch.nn.Module, data_loader: DataLoader, criterion: torch.nn.Module,
            reconstruction: bool, num_samples: int, device: str) -> List[float]:
    """
    Makes predictions using a Pytorch model and returns evaluation metrics.

    Args:
        model (torch.nn.Module): The PyTorch model for making predictions.
        data_loader (torch.utils.data.DataLoader): The DataLoader for making predictions.
        criterion (torch.nn.Module): The loss function to consider for evaluation.
        reconstruction (bool): Whether the model is used for reconstruction or classification. Defaults to True.
        num_samples (bool): The number of samples used to calculating lift and demotion scores.
                            Defaults to 100. When the reconstruction flag is set to True, this parameter
                            is not used.
        device (str): The device to use for predictions.

    Returns:
        list: A list of evaluation metrics.
    """
    model.to(device)  # move the model to the specified device
    model.eval()  # set the mod-1el to evaluation mode

    # initialize vars to store true labels, predicted labels, and predicted scores
    y_true, y_pred, y_scores = [], [], []
    # initialize vars to store data, outputs, and losses
    losses, data, outputs = [], [], []

    with torch.no_grad():  # disable gradient computation
        for x, y in data_loader:  # loop through batches
            # append the original data
            data.append(x)
            y_true.append(y)
            # move the input features and targets to the specified device
            x = x.to(device)
            y = y.to(device)
            # make predictions
            out = model(x)
            # calculate loss
            loss = criterion(out, x) if reconstruction else criterion(out, y.reshape(-1, 1))

            # Convert model outputs to predicted scores and labels
            predicted_scores = torch.sigmoid(out.cpu())
            predicted = torch.round(predicted_scores)
            y_pred.append(predicted)
            y_scores.append(predicted_scores)

            # append predictions
            outputs.append(out.detach().cpu())
            # append batch loss
            losses.append(loss.item())

    if reconstruction:  # calculate the loss for the reconstruction task
        loss = sum(losses) / len(losses)
        return [loss]

    # concatenate true and predicted labels and scores
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_scores = torch.cat(y_scores, dim=0)

    # calculate classification metrics
    acc, precision, recall, f1 = get_classification_metrics(y_true.numpy(), y_pred.numpy())
    brier, roc_auc, pr_auc = get_probability_measures(y_true.numpy(), y_scores.numpy())

    # calculate lift, demotion and the weighted scores
    s1 = int(0.1 * len(y_scores))
    s2 = int(0.2 * len(y_scores))
    s3 = int(0.3 * len(y_scores))
    s4 = int(0.4 * len(y_scores))
    s5 = int(0.5 * len(y_scores))
    s6 = int(0.6 * len(y_scores))
    s7 = int(0.7 * len(y_scores))
    s8 = int(0.8 * len(y_scores))
    s9 = int(0.9 * len(y_scores))
    s10 = int(1. * len(y_scores))

    tt = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]

    for t in tt:
        lift, demotion, weighted = get_lift_demotion_scores(y_true.numpy(), y_scores.numpy(),
                                                            num_samples=t,
                                                            verbose=False)
        print(f"Lift:{lift}, demotion:{demotion}, weighted:{weighted}")

    aux = [loss, acc,
           precision, recall, f1, brier, roc_auc, pr_auc, lift, demotion, weighted]
    return aux
