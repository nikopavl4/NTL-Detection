from typing import Tuple, Optional

import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             accuracy_score, average_precision_score, brier_score_loss)


def get_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Generates a simple classification report by computing the precision, recall and f1 scores.

    Args:
        y_true (np.ndarray): The array of true labels.
        y_pred (np.ndarray): The array of predicted labels.

    Returns:
        precision (float): The precision score.
        recall (flat): the recall score.
        f1 (float): the f1 score.
        roc_auc (None or float): the roc_auc score. It can be None if y_score is not provided.
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return acc, precision, recall, f1


def get_probability_measures(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, float]:
    """
    Generates a report by using the ground truth labels and the predicted probabilities.

    Args:
        y_true (np.ndarray): The array of true labels.
        y_scores (np.ndarray): The array of scores.

    Returns:
        brier (float): the brier score loss.
        roc_auc (float): the roc_auc score.
        pr_auc (flat): the average precision score.
    """
    try:
        brier = brier_score_loss(y_true, y_scores)
    except ValueError:
        brier = None
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = None
    try:
        pr_auc = average_precision_score(y_true, y_scores)
    except ValueError:
        pr_auc = None
    return brier, roc_auc, pr_auc


def find_all(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """
    Finds all positive samples (NTLs) by iteratively considering subsets of samples sorted by their predicted scores
    until all NTLs have been found.

    Args:
        y_true (np.ndarray): The array of true labels.
        y_scores (np.ndarray): The array of scores.
    """
    target = len(y_true[y_true == 1])  # number of positive samples
    total_samples = len(y_true)  # total number of samples
    print(f"There are {target} ones out of {total_samples} samples.")

    # indices that sort y_scores
    sorted_scores = np.argsort(y_scores.reshape(1, -1)[0])

    # subsets of samples in steps of 100
    steps = np.arange(100, total_samples + 100, 100)
    steps_to_print = [100]
    tmp = np.arange(500, total_samples + 500, 500)
    steps_to_print.extend(list(tmp))
    print(f"Finding all ones with a maximum of {len(steps)} steps...")
    for step in steps:
        # select the last num_samples indices (the highest scores)
        top_indices_lift = sorted_scores[-step:]
        # get scores and labels for the selected samples
        tmp_scores = y_scores[top_indices_lift]
        # count the number of positive samples found
        y_true_lift = y_true[top_indices_lift]
        num_found = len(y_true_lift[y_true_lift == 1])
        # print progress
        if step in steps_to_print:
            print(f"Found {num_found} out of {step} samples considered (sample scores: {tmp_scores[:10]}...")
        if num_found == target:
            print(f"Found all at step {step}! ({num_found} out of {target}")
            break


def get_lift_demotion_scores(y_true: np.ndarray,
                             y_scores: np.ndarray,
                             num_samples: Optional[int] = 100,
                             verbose: Optional[bool] = False) -> Tuple[float, float, float]:
    """
    Computes the Lift and Demotion scores as specified by the organizers.

    The Lift score measures how well the model identify NTLs.
    To calculate the lift score, we get the top num_samples (defaults to 100) samples with the highest probability
    of being NTLs. Then, we calculate how many of these samples are actually NTLs.

    The Demotion score measures how well the model identify non-NTL samples.
    To calculate the demotion score, we get the top num_samples (defaults to 100) samples with the lowest probability
    of being non-NTLs. Then, we calculate how many of these samples are actually non-NTLs.

    Args:
        y_true (np.ndarray): The array of true labels.
        y_scores (np.ndarray): The array of scores.
        num_samples (int): The number of samples to consider for calculating the Lift and Demotion scores.
        verbose (bool): Whether to allow verbosing or not

    Returns:
        lift_score (float): The lift score.
        demotion_score (float): The demotion score.
        weighted_score (float): The weighted lift and demotion score.
    """

    if num_samples < 1:
        print("[Warning] Cannot operate with less than 1 sample! Please increase the number of samples to consider!")
        return -1., -1., -1.

    # indices that sort y_scores
    sorted_scores = np.argsort(y_scores.reshape(1, -1)[0])

    # select the last num_samples indices (the highest scores)
    top_indices_lift = sorted_scores[-num_samples:]
    # select the first num_samples indices (the lowest scores)
    top_indices_demotion = sorted_scores[:num_samples]

    if verbose:
        print(f"Lift indices: {top_indices_lift}")
        print(f"Demotion indices: {top_indices_demotion}")

    # get the y_true found in top and lowest scores for calculating lift and demotion scores
    y_true_lift = y_true[top_indices_lift]
    y_true_demotion = y_true[top_indices_demotion]

    if verbose:
        print("Bincount in y_true found in Lift:", np.bincount(y_true_lift))
        print("Bincount in y_true found in Demotion:", np.bincount(y_true_demotion))

    lift_score = accuracy_score(np.ones_like(y_true_lift), y_true_lift)
    demotion_score = accuracy_score(np.zeros_like(y_true_demotion), y_true_demotion)
    weighted_score = 0.7 * lift_score + 0.3 * demotion_score

    return lift_score, demotion_score, weighted_score
