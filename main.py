import sys

import torch
from torch import nn

# from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes

from common.args import parse_args
from common.utils import get_optimizer, load_model
from data import pretrain_dataset
from data.dataset import get_meta_dataset
from models.joing_embeddings_models import EncoderF
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


### PRETRAIN ###

def pretrain(P):
    from torch.utils.data import DataLoader
    from data.pretrain_dataset import PretrainDataset
    from models.joing_embeddings_models import EncoderF, ProjectorP
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PretrainDataset(f'./data/{P.dataset}/train_x.npy', mask_ratio=0.2)
    loader = DataLoader(dataset, batch_size=P.batch_size, shuffle=True, num_workers=4)

    f = EncoderF(input_dim=dataset.D).to(device)
    p = ProjectorP(embed_dim=256, input_dim=dataset.D).to(device)
    optimizer = torch.optim.Adam(list(f.parameters()) + list(p.parameters()), lr=P.lr)

    for epoch in range(100):
        total_loss = 0.0
        valid_batches = 0

        for batch in loader:
            x_input = batch['x_input'].to(device)  # x[S']
            x_full = batch['x_full'].to(device)  # original x
            mask = batch['mask'].to(device)  # binary mask

            # Forward pass
            z = f(x_input)
            h = p(z, mask)
            h = F.normalize(h, dim=1, eps=1e-8)

            x_hidden = x_full * (1 - mask)
            x_hidden = F.normalize(x_hidden, dim=1, eps=1e-8)

            sim = torch.matmul(x_hidden, x_hidden.t())

            sim = sim - torch.diag(torch.ones(sim.size(0), device=device) * float('inf'))

            valid_rows = (x_hidden.abs().sum(dim=1) > 1e-6)
            invalid_rows = (valid_rows == False)

            sim = sim.masked_fill(invalid_rows.unsqueeze(1), -float('inf'))
            sim = sim.masked_fill(invalid_rows.unsqueeze(0), -float('inf'))

            pos_idx = torch.argmax(sim, dim=1)
            max_sim = torch.gather(sim, 1, pos_idx.unsqueeze(1)).squeeze()
            valid_pairs = (max_sim > -float('inf')) & valid_rows

            if not valid_pairs.any():
                continue

            logits = torch.matmul(h, h.t()) / 0.1
            logits.fill_diagonal_(-float('inf'))

            loss = F.cross_entropy(logits[valid_pairs], pos_idx[valid_pairs])

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(f.parameters()) + list(p.parameters()), 1.0)
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

        # Print epoch stats
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.6f}")
        else:
            print(f"[Epoch {epoch + 1}] Warning: No valid batches")

    torch.save({'f': f.state_dict(), 'p': p.state_dict()}, 'f_p_pretrained.pt')
    print("SAVED")


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

    """ pretraining """
    print("Starting pretraining...")
    pretrain(P)
    print("Ended pretraining")

    """ define dataset and dataloader """
    # kwargs = {'batch_size': P.batch_size, 'shuffle': True,
    #           'pin_memory': True, 'num_workers': 2}
    train_set, val_set, test_set = get_meta_dataset(P, dataset=P.dataset)

    train_loader = train_set
    test_loader = val_set

    """ Initialize model, optimizer, loss_scalar (for amp) and scheduler """
    #model = get_model(P, P.model).to(device)
    model = EncoderF(input_dim=train_set.tabular_size).to(device)
    checkpoint = torch.load('f_p_pretrained.pt')
    model.load_state_dict(checkpoint['f'])
    model.to(device)
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
    # load_model(P, model, logger)

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
