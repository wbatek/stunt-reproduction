import torch
from torchvision import transforms

from data.cmc import Cmc
from data.dna import Dna
# from torchmeta.transforms import ClassSplitter, Categorical

from data.income import Income
from data.diabetes import Diabetes
from data.karhunen import Karhunen
from data.optdigits import Optdigits
from data.pixel import Pixel
from train.trainer import meta_trainer
from data.semeion import Semeion


def get_meta_dataset(P, dataset, eps, only_test=False):
    if dataset == 'income':
        meta_train_dataset = Income(P,
                                    tabular_size=105,
                                    seed=P.seed,
                                    source='train',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size,
                                    test_num_way=P.num_ways,
                                    query=P.num_shots_test,
                                    eps=eps)

        meta_val_dataset = Income(P,
                                  tabular_size=105,
                                  seed=P.seed,
                                  source='val',
                                  shot=1,
                                  tasks_per_batch=P.test_batch_size,
                                  test_num_way=2,
                                  query=30,
                                  eps=eps)

        meta_test_dataset = Income(P,
                                   tabular_size=105,
                                   seed=P.seed,
                                   source='test',
                                   shot=P.num_shots,
                                   tasks_per_batch=10,
                                   test_num_way=2,
                                   query=5,
                                   eps=eps)
    elif dataset == 'diabetes':
        meta_train_dataset = Diabetes(P,
                                      tabular_size=8,
                                      seed=P.seed,
                                      source='train',
                                      shot=P.num_shots,
                                      tasks_per_batch=P.batch_size,
                                      test_num_way=P.num_ways,
                                      query=P.num_shots_test,
                                      eps=eps)

        meta_val_dataset = Diabetes(P,
                                    tabular_size=8,
                                    seed=P.seed,
                                    source='val',
                                    shot=1,
                                    tasks_per_batch=P.test_batch_size,
                                    test_num_way=2,
                                    query=30,
                                    eps=eps)

        meta_test_dataset = Diabetes(P,
                                     tabular_size=8,
                                     seed=P.seed,
                                     source='test',
                                     shot=P.num_shots,
                                     tasks_per_batch=10,
                                     test_num_way=2,
                                     query=5,
                                     eps=eps)

    elif dataset == 'optdigits':
        meta_train_dataset = Optdigits(P,
                                       tabular_size=64,
                                       seed=P.seed,
                                       source='train',
                                       shot=P.num_shots,
                                       tasks_per_batch=P.batch_size,
                                       test_num_way=P.num_ways,
                                       query=P.num_shots_test)

        meta_val_dataset = Optdigits(P,
                                     tabular_size=64,
                                     seed=P.seed,
                                     source='val',
                                     shot=1,
                                     tasks_per_batch=P.test_batch_size,
                                     test_num_way=2,
                                     query=30)

        meta_test_dataset = Optdigits(P,
                                      tabular_size=64,
                                      seed=P.seed,
                                      source='test',
                                      shot=P.num_shots,
                                      tasks_per_batch=10,
                                      test_num_way=P.num_ways,
                                      query=5)
    elif dataset == 'cmc':
        meta_train_dataset = Cmc(P,
                                tabular_size=9,
                                seed=P.seed,
                                source='train',
                                shot=P.num_shots,
                                tasks_per_batch=P.batch_size,
                                test_num_way=P.num_ways,
                                query=P.num_shots_test)
        meta_val_dataset = Cmc(P,
                               tabular_size=9,
                              seed=P.seed,
                              source='val',
                              shot=1,
                              tasks_per_batch=P.test_batch_size,
                              test_num_way=2,
                              query=30)
        meta_test_dataset = Cmc(P,
                                tabular_size=9,
                                seed=P.seed,
                                source='test',
                                shot=P.num_shots,
                                tasks_per_batch=10,
                                test_num_way=P.num_ways,
                                query=5)
    elif dataset == 'dna':
        meta_train_dataset = Dna(P,
                                tabular_size=180,
                                seed=P.seed,
                                source='train',
                                shot=P.num_shots,
                                tasks_per_batch=P.batch_size,
                                test_num_way=P.num_ways,
                                query=P.num_shots_test,
                                eps=0)
        meta_val_dataset = Dna(P,
                                tabular_size=180,
                                seed=P.seed,
                                source='val',
                                shot=1,
                                tasks_per_batch=P.test_batch_size,
                                test_num_way=2,
                                query=30,
                                eps=0)
        meta_test_dataset = Dna(P,
                                tabular_size=180,
                                seed=P.seed,
                                source='test',
                                shot=P.num_shots,
                                tasks_per_batch=10,
                                test_num_way=P.num_ways,
                                query=5,
                                eps=0)
    elif dataset == 'karhunen':
        meta_train_dataset = Karhunen(P,
                                             tabular_size=64,
                                             seed=P.seed,
                                             source='train',
                                             shot=P.num_shots,
                                             tasks_per_batch=P.batch_size,
                                             test_num_way=P.num_ways,
                                             query=P.num_shots_test)
        meta_val_dataset = Karhunen(P,
                                    tabular_size=64,
                                    seed=P.seed,
                                    source='val',
                                    shot=1,
                                    tasks_per_batch=P.test_batch_size,
                                    test_num_way=2,
                                    query=30)
        meta_test_dataset = Karhunen(P,
                                     tabular_size=64,
                                     seed=P.seed,
                                     source='test',
                                     shot=P.num_shots,
                                     tasks_per_batch=10,
                                     test_num_way=P.num_ways,
                                     query=5)
    elif dataset == 'semeion':
        meta_train_dataset = Semeion(P,
                                    tabular_size=256,
                                    seed=P.seed,
                                    source='train',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size,
                                    test_num_way=P.num_ways,
                                    query=P.num_shots_test)
        meta_val_dataset = Semeion(P,
                                  tabular_size=256,
                                  seed=P.seed,
                                  source='val',
                                  shot=1,
                                  tasks_per_batch=P.test_batch_size,
                                  test_num_way=2,
                                  query=30)
        meta_test_dataset = Semeion(P,
                                   tabular_size=256,
                                   seed=P.seed,
                                   source='test',
                                   shot=P.num_shots,
                                   tasks_per_batch=10,
                                   test_num_way=P.num_ways,
                                      query=5)
    elif dataset == 'pixel':
        meta_train_dataset = Pixel(P,
                                    tabular_size=240,
                                    seed=P.seed,
                                    source='train',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size,
                                    test_num_way=P.num_ways,
                                    query=P.num_shots_test)
        meta_val_dataset = Pixel(P,
                                  tabular_size=240,
                                  seed=P.seed,
                                  source='val',
                                  shot=1,
                                  tasks_per_batch=P.test_batch_size,
                                  test_num_way=2,
                                  query=30)
        meta_test_dataset = Pixel(P,
                                   tabular_size=240,
                                   seed=P.seed,
                                   source='test',
                                   shot=P.num_shots,
                                   tasks_per_batch=10,
                                   test_num_way=P.num_ways,
                                   query=5)


    else:
        raise NotImplementedError()

    return meta_train_dataset, meta_val_dataset, meta_test_dataset
