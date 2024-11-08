import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

def tf_attack(logits_train, logits_test, loss_train, loss_test, train_labels, test_labels):
    attack_input = AttackInputData(
    logits_train = logits_train,
    logits_test = logits_test,
    loss_train = loss_train,
    loss_test = loss_test,
    labels_train = train_labels,
    labels_test = test_labels
    )

    slicing_spec = SlicingSpec(
        entire_dataset = True,
        by_class = False,
        by_percentiles = False,
    by_classification_correctness = False
    )

    attack_types = [
        AttackType.THRESHOLD_ATTACK,
        # AttackType.LOGISTIC_REGRESSION,
        # AttackType.RANDOM_FOREST,
        # AttackType.K_NEAREST_NEIGHBORS,
        # AttackType.THRESHOLD_ENTROPY_ATTACK
    ] 

    attacks_result = mia.run_attacks(attack_input=attack_input,
                                    slicing_spec=slicing_spec,
                                 attack_types=attack_types)
    return attacks_result



import numpy as np
from torch import nn
import torch

def compute_attack_components(net, loader, device='cuda'):
    net.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    all_logits = []
    all_labels = []
    all_predicted_labels = []

    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        logits = net(features)
        # Compute losses
        losses = criterion(logits, targets).detach().cpu().numpy()
        # Append losses to the list
        all_losses.extend(losses)

        # Append logits and labels to their respective lists
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(targets.detach().cpu().numpy())

        # Compute predicted labels
        predicted_labels = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_predicted_labels.append(predicted_labels)

    # Concatenate logits, labels, and predicted labels along the samples axis
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_predicted_labels = np.concatenate(all_predicted_labels, axis=0)

    return np.array(all_logits), np.array(all_losses), np.array(all_labels), np.array(all_predicted_labels)

