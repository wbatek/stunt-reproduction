import sys

import torch
from tabpfn import TabPFNClassifier

# from torchmeta.utils.data import BatchMetaDataLoader
from common.args import parse_args
from common.utils import get_optimizer, load_model
from data.dataset import get_meta_dataset
from models.model import get_model
from train.trainer import meta_trainer
from utils import Logger, set_random_seed
import numpy as np

from data.dataset import dataset_to_tabular_size


def get_accuracy(prototypes, test_embeddings, test_targets):
    """
    Compute the accuracy of predictions based on the prototypes.

    Parameters:
    - prototypes: Tensor of shape (num_ways, embedding_dim)
                  Representing the class prototypes.
    - test_embeddings: Tensor of shape (num_test_samples, embedding_dim)
                       Embeddings of the test samples.
    - test_targets: Tensor of shape (num_test_samples,)
                    Ground truth class labels for the test samples.

    Returns:
    - accuracy: Float Tensor representing the classification accuracy.
    """
    # Compute squared distances between test embeddings and prototypes
    squared_distances = torch.sum((prototypes.unsqueeze(2)
                                   - test_embeddings.unsqueeze(1)) ** 2, dim=-1)  # Shape: (num_ways, num_test_samples)

    # Get the predicted classes (closest prototype)
    predicted_classes = torch.argmin(squared_distances, dim=1)  # Shape: (num_test_samples,)
    # Compare predictions with ground truth and compute accuracy
    correct = (predicted_classes == test_targets).sum().float()
    accuracy = correct / test_targets.size(1)

    return accuracy


def test(P, model, criterion, logger, test_set):
    accuracies = []

    total_accuracy = 0
    total_loss = 0
    total_tasks = 0

    input_dim = dataset_to_tabular_size[P.dataset]
    max_dim = P.max_dim
    num_submodels = P.num_submodels

    feature_subsets = [np.random.choice(input_dim, max_dim, replace=False) for _ in range(num_submodels)]

    with torch.no_grad():
        for i in range(P.outer_steps):
            print(i)
            batch = test_set.get_test_batch()

            for task in batch:
                support_inputs, support_targets = task['train']
                query_inputs, query_targets = task['test']
                support_inputs = support_inputs.squeeze(0).numpy()  # Shape: (num_way * shot, tabular_size)
                query_inputs = query_inputs.squeeze(0).numpy()  # Shape: (num_way * query, tabular_size)
                support_targets = support_targets.squeeze(0).numpy()  # Shape: (num_way * shot,)
                query_targets = query_targets.squeeze(0).numpy()  # Shape: (num_way * query,)

                all_preds = []

                for feature_idx in feature_subsets:
                    model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
                    model.fit(support_inputs[:, feature_idx], support_targets)
                    preds = model.predict(query_inputs[:, feature_idx])
                    all_preds.append(preds)

                all_preds = np.array(all_preds).T

                final_preds = []
                for pred_row in all_preds:
                    counts = np.bincount(pred_row)
                    final_preds.append(np.argmax(counts))

                #model.fit(support_inputs.numpy(), support_targets.numpy())

                #y_eval = model.predict(query_inputs.numpy())

                # correct = sum(1 for i in range(len(y_eval)) if y_eval[i] == query_targets.numpy()[i])
                # acc = correct / len(y_eval)
                # total_accuracy += acc
                # total_tasks += 1
                correct = sum(1 for i in range(len(final_preds)) if final_preds[i] == query_targets[i])
                acc = correct / len(query_targets)
                total_accuracy += acc
                total_tasks += 1
            if i % 100 == 0:
                avg_accuracy = total_accuracy / total_tasks
                print(f"Step {i}, Average accuracy: {avg_accuracy:.4f}")
                logger.log(f"Step {i}, Avg Accuracies = {avg_accuracy:.4f}")


    avg_accuracy = total_accuracy / total_tasks
    avg_loss = total_loss / total_tasks
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")

    # logger.scalar_summary('test/accuracy', avg_accuracy, 0)
    # logger.scalar_summary('test/loss', avg_loss, 0)
    # logger.log(avg_accuracy)
    # logger.log(avg_loss)
    return avg_accuracy


def main(rank, P):
    P.rank = rank

    """ set torch device"""
    if torch.cuda.is_available():
        torch.cuda.set_device(P.rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    """ fixing randomness """
    set_random_seed(P.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    """ define dataset and dataloader """
    # kwargs = {'batch_size': P.batch_size, 'shuffle': True,
    #           'pin_memory': True, 'num_workers': 2}
    train_set, val_set, test_set = get_meta_dataset(P, dataset=P.dataset)

    train_loader = train_set
    test_loader = val_set

    """ Initialize model, optimizer, loss_scalar (for amp) and scheduler """
    model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    # optimizer = get_optimizer(P, model)

    """ define train and test type """
    from train import setup as train_setup
    from evals import setup as test_setup
    train_func, fname, today = train_setup(P.mode, P)
    test_func = test_setup(P.mode, P)

    """ define logger """
    logger = Logger(fname, ask=P.resume_path is None, today=today, rank=P.rank)
    logger.log(P)
    logger.log(model)

    """ load model if necessary """
    load_model(P, model, logger)

    """ train """
    #meta_trainer(P, train_func, test_func, model, train_loader, test_loader, logger)
    """ test """
    criterion = nn.CrossEntropyLoss()

    avg_acc = test(P, model, criterion, logger, test_set)
    """ close tensorboard """
    logger.close_writer()


if __name__ == "__main__":
    import os
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly",
        category=UserWarning
    )
    import torch
    from torch import nn

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    """ argument define """
    P = parse_args()

    P.world_size = torch.cuda.device_count()
    P.distributed = P.world_size > 1
    if P.distributed:
        print("currently, ddp is not supported, should consider transductive BN before using ddp",
              file=sys.stderr)
    else:
        main(0, P)
