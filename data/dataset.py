import importlib

import torch
from torchvision import transforms

from data.dataset_impl import Dataset


dataset_to_num_classes = {
    'income': 2,
    'diabetes': 2,
    'optdigits': 10,
    'cmc': 3,
    'dna': 3,
    'karhunen': 10,
    'semeion': 10,
    'pixel': 10
}

dataset_to_tabular_size = {
    'income': 105,
    'diabetes': 8,
    'optdigits': 64,
    'cmc': 24,
    'dna': 180,
    'karhunen': 64,
    'semeion': 256,
    'pixel': 240
}

def get_meta_dataset(P, dataset):
    try:
        module_path = f"data.{dataset}.retrieve_files"  # Example: 'diabetes.retrieve_files'
        retrieve_module = importlib.import_module(module_path)
        retrieve_function = getattr(retrieve_module, "retrieve")
        retrieve_function(P.seed)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not import retrieve function for dataset '{dataset}': {e}")


    meta_train_dataset = Dataset(P,
                                     name=dataset,
                                     tabular_size=dataset_to_tabular_size[dataset],
                                     seed=P.seed,
                                     source='train',
                                     shot=P.num_shots,
                                     tasks_per_batch=P.batch_size,
                                     test_num_way=P.num_ways,
                                     query=P.num_shots_test,
                                     num_classes=dataset_to_num_classes[dataset])

    meta_val_dataset = Dataset(P,
                                   name=dataset,
                                   tabular_size=dataset_to_tabular_size[dataset],
                                   seed=P.seed,
                                   source='val',
                                   shot=1,
                                   tasks_per_batch=P.test_batch_size,
                                   test_num_way=2,
                                   query=15,
                                   num_classes=dataset_to_num_classes[dataset])

    meta_test_dataset = Dataset(P,
                                    name=dataset,
                                    tabular_size=dataset_to_tabular_size[dataset],
                                    seed=P.seed,
                                    source='test',
                                    shot=P.num_shots,
                                    tasks_per_batch=10,
                                    test_num_way=2,
                                    query=5,
                                    num_classes=dataset_to_num_classes[dataset])
    # if dataset == 'income':
    #     meta_train_dataset = Dataset(P,
    #                                  name=dataset,
    #                                  tabular_size=dataset_to_tabular_size[dataset],
    #                                  seed=P.seed,
    #                                  source='train',
    #                                  shot=P.num_shots,
    #                                  tasks_per_batch=P.batch_size,
    #                                  test_num_way=P.num_ways,
    #                                  query=P.num_shots_test,
    #                                  num_classes=dataset_to_num_classes[dataset])
    #
    #     meta_val_dataset = Dataset(P,
    #                                name=dataset,
    #                                tabular_size=dataset_to_tabular_size[dataset],
    #                                seed=P.seed,
    #                                source='val',
    #                                shot=1,
    #                                tasks_per_batch=P.test_batch_size,
    #                                test_num_way=2,
    #                                query=30,
    #                                num_classes=dataset_to_num_classes[dataset])
    #
    #     meta_test_dataset = Dataset(P,
    #                                 name=dataset,
    #                                 tabular_size=dataset_to_tabular_size[dataset],
    #                                 seed=P.seed,
    #                                 source='test',
    #                                 shot=P.num_shots,
    #                                 tasks_per_batch=10,
    #                                 test_num_way=2,
    #                                 query=5,
    #                                 num_classes=dataset_to_num_classes[dataset])
    # elif dataset == 'diabetes':
    #     meta_train_dataset = Dataset(P,
    #                                  name='diabetes',
    #                                  tabular_size=8,
    #                                  seed=P.seed,
    #                                  source='train',
    #                                  shot=P.num_shots,
    #                                  tasks_per_batch=P.batch_size,
    #                                  test_num_way=P.num_ways,
    #                                  query=P.num_shots_test,
    #                                  num_classes=2)
    #
    #     meta_val_dataset = Dataset(P,
    #                                name='diabetes',
    #                                tabular_size=8,
    #                                seed=P.seed,
    #                                source='val',
    #                                shot=1,
    #                                tasks_per_batch=P.test_batch_size,
    #                                test_num_way=2,
    #                                query=30,
    #                                num_classes=2)
    #
    #     meta_test_dataset = Dataset(P,
    #                                 name='diabetes',
    #                                 tabular_size=8,
    #                                 seed=P.seed,
    #                                 source='test',
    #                                 shot=P.num_shots,
    #                                 tasks_per_batch=10,
    #                                 test_num_way=2,
    #                                 query=5,
    #                                 num_classes=2)
    #
    # elif dataset == 'optdigits':
    #     meta_train_dataset = Dataset(P,
    #                                     name='optdigits',
    #                                    tabular_size=64,
    #                                    seed=P.seed,
    #                                    source='train',
    #                                    shot=P.num_shots,
    #                                    tasks_per_batch=P.batch_size,
    #                                    test_num_way=P.num_ways,
    #                                    query=P.num_shots_test,
    #                                    num_classes=10)
    #
    #     meta_val_dataset = Dataset(P,
    #                                  tabular_size=64,
    #                                  seed=P.seed,
    #                                  source='val',
    #                                  shot=1,
    #                                  tasks_per_batch=P.test_batch_size,
    #                                  test_num_way=2,
    #                                  query=30)
    #
    #     meta_test_dataset = Optdigits(P,
    #                                   tabular_size=64,
    #                                   seed=P.seed,
    #                                   source='test',
    #                                   shot=P.num_shots,
    #                                   tasks_per_batch=10,
    #                                   test_num_way=P.num_ways,
    #                                   query=5)
    # elif dataset == 'cmc':
    #     meta_train_dataset = Cmc(P,
    #                              tabular_size=24,
    #                              seed=P.seed,
    #                              source='train',
    #                              shot=P.num_shots,
    #                              tasks_per_batch=P.batch_size,
    #                              test_num_way=P.num_ways,
    #                              query=P.num_shots_test)
    #     meta_val_dataset = Cmc(P,
    #                            tabular_size=24,
    #                            seed=P.seed,
    #                            source='val',
    #                            shot=1,
    #                            tasks_per_batch=P.test_batch_size,
    #                            test_num_way=2,
    #                            query=30)
    #     meta_test_dataset = Cmc(P,
    #                             tabular_size=24,
    #                             seed=P.seed,
    #                             source='test',
    #                             shot=P.num_shots,
    #                             tasks_per_batch=10,
    #                             test_num_way=P.num_ways,
    #                             query=5)
    # elif dataset == 'dna':
    #     meta_train_dataset = Dna(P,
    #                              tabular_size=180,
    #                              seed=P.seed,
    #                              source='train',
    #                              shot=P.num_shots,
    #                              tasks_per_batch=P.batch_size,
    #                              test_num_way=P.num_ways,
    #                              query=P.num_shots_test,
    #                              eps=0)
    #     meta_val_dataset = Dna(P,
    #                            tabular_size=180,
    #                            seed=P.seed,
    #                            source='val',
    #                            shot=1,
    #                            tasks_per_batch=P.test_batch_size,
    #                            test_num_way=2,
    #                            query=30,
    #                            eps=0)
    #     meta_test_dataset = Dna(P,
    #                             tabular_size=180,
    #                             seed=P.seed,
    #                             source='test',
    #                             shot=P.num_shots,
    #                             tasks_per_batch=10,
    #                             test_num_way=P.num_ways,
    #                             query=5,
    #                             eps=0)
    # elif dataset == 'karhunen':
    #     meta_train_dataset = Karhunen(P,
    #                                   tabular_size=64,
    #                                   seed=P.seed,
    #                                   source='train',
    #                                   shot=P.num_shots,
    #                                   tasks_per_batch=P.batch_size,
    #                                   test_num_way=P.num_ways,
    #                                   query=P.num_shots_test)
    #     meta_val_dataset = Karhunen(P,
    #                                 tabular_size=64,
    #                                 seed=P.seed,
    #                                 source='val',
    #                                 shot=1,
    #                                 tasks_per_batch=P.test_batch_size,
    #                                 test_num_way=2,
    #                                 query=30)
    #     meta_test_dataset = Karhunen(P,
    #                                  tabular_size=64,
    #                                  seed=P.seed,
    #                                  source='test',
    #                                  shot=P.num_shots,
    #                                  tasks_per_batch=10,
    #                                  test_num_way=P.num_ways,
    #                                  query=5)
    # elif dataset == 'semeion':
    #     meta_train_dataset = Semeion(P,
    #                                  tabular_size=256,
    #                                  seed=P.seed,
    #                                  source='train',
    #                                  shot=P.num_shots,
    #                                  tasks_per_batch=P.batch_size,
    #                                  test_num_way=P.num_ways,
    #                                  query=P.num_shots_test)
    #     meta_val_dataset = Semeion(P,
    #                                tabular_size=256,
    #                                seed=P.seed,
    #                                source='val',
    #                                shot=1,
    #                                tasks_per_batch=P.test_batch_size,
    #                                test_num_way=2,
    #                                query=30)
    #     meta_test_dataset = Semeion(P,
    #                                 tabular_size=256,
    #                                 seed=P.seed,
    #                                 source='test',
    #                                 shot=P.num_shots,
    #                                 tasks_per_batch=10,
    #                                 test_num_way=P.num_ways,
    #                                 query=5)
    # elif dataset == 'pixel':
    #     meta_train_dataset = Pixel(P,
    #                                tabular_size=240,
    #                                seed=P.seed,
    #                                source='train',
    #                                shot=P.num_shots,
    #                                tasks_per_batch=P.batch_size,
    #                                test_num_way=P.num_ways,
    #                                query=P.num_shots_test)
    #     meta_val_dataset = Pixel(P,
    #                              tabular_size=240,
    #                              seed=P.seed,
    #                              source='val',
    #                              shot=1,
    #                              tasks_per_batch=P.test_batch_size,
    #                              test_num_way=2,
    #                              query=30)
    #     meta_test_dataset = Pixel(P,
    #                               tabular_size=240,
    #                               seed=P.seed,
    #                               source='test',
    #                               shot=P.num_shots,
    #                               tasks_per_batch=10,
    #                               test_num_way=P.num_ways,
    #                               query=5)

    return meta_train_dataset, meta_val_dataset, meta_test_dataset
