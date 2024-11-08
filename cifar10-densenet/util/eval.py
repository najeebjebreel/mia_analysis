from __future__ import print_function, absolute_import

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Applies the softmax function to an array of logits.

    Args:
    logits (np.ndarray): Array of logits with shape (n_samples, n_classes).

    Returns:
    np.ndarray: Array of probabilities with the same shape as logits.
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def compute_accuracy(logits: np.ndarray, true_labels: np.ndarray):
    """
    Computes the accuracy given logits and true labels.

    Args:
    logits (np.ndarray): Array of logits with shape (n_samples, n_classes).
    true_labels (np.ndarray): Array of true labels with shape (n_samples,).

    Returns:
    float: Accuracy of the predictions.
    """
    # Apply softmax to convert logits to probabilities
    probabilities = softmax(logits)
    
    # Convert probabilities to predicted labels by taking the argmax
    predicted_labels = np.argmax(probabilities, axis=1)
    
    # Calculate the number of correct predictions
    correct_predictions = np.sum(predicted_labels == true_labels)
    
    # Compute accuracy
    accuracy = correct_predictions / len(true_labels)
    
    return accuracy, probabilities
