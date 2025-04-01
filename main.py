import sys

import torch
from torch import nn

# from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes

from common.args import parse_args
from common.utils import get_optimizer, load_model
from data.dataset import get_meta_dataset
from data.income import Income
from models.model import get_model
from train.trainer import meta_trainer
from utils import Logger, set_random_seed


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


def test(P, model, optimizer, criterion, logger, test_set):
    accuracies = []

    total_accuracy = 0
    total_loss = 0
    total_tasks = 0

    model.eval()

    with torch.no_grad():
        for i in range(P.outer_steps):
            print(i)
            batch = test_set.get_test_batch()

            for task in batch:
                support_inputs, support_targets = task['train']
                query_inputs, query_targets = task['test']
                support_embeddings = model(support_inputs)
                query_embeddings = model(query_inputs)

                prototypes = get_prototypes(support_embeddings, support_targets, test_set.num_classes)

                squared_distances = torch.sum((prototypes.unsqueeze(2) - query_embeddings.unsqueeze(1)) ** 2, dim=-1)

                loss = criterion(-squared_distances, query_targets)

                acc = get_accuracy(prototypes, query_embeddings, query_targets).item()
                accuracies.append(acc)
                total_accuracy += acc
                total_loss += loss.item()
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
    logger.log(avg_accuracy)
    logger.log(avg_loss)
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
    model = get_model(P, P.model).to(device)
    optimizer = get_optimizer(P, model)

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
    meta_trainer(P, train_func, test_func, model, optimizer, train_loader, test_loader, logger)
    """ test """
    criterion = nn.CrossEntropyLoss()

    avg_acc = test(P, model, optimizer, criterion, logger, test_set)
    """ close tensorboard """
    logger.close_writer()


if __name__ == "__main__":
    import os

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
