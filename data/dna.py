from re import I
import numpy as np
import torch
import os
import copy
import faiss
from sklearn.mixture import GaussianMixture


class Dna(object):
    def __init__(self, P, tabular_size, seed, source, shot, tasks_per_batch, test_num_way, query, eps):
        super().__init__()
        self.num_classes = 3
        self.tabular_size = 180
        self.source = source
        self.shot = shot
        self.query = query
        self.tasks_per_batch = tasks_per_batch
        self.unlabeled_x = np.load('./data/dna/train_x.npy', allow_pickle=True)
        self.test_x = np.load('./data/dna/xtest.npy', allow_pickle=True)
        self.test_y = np.load('./data/dna/ytest.npy', allow_pickle=True)
        self.val_x = np.load('./data/dna/val_x.npy', allow_pickle=True)
        self.val_y = np.load(
            './data/dna/yval.npy', allow_pickle=True)  # val_y is given from pseudo-validaiton scheme with STUNT
        self.test_num_way = test_num_way
        self.test_rng = np.random.RandomState(seed)
        self.val_rng = np.random.RandomState(seed)
        self.invalid_count = 0
        self.eps = eps

    def __next__(self):
        return self.get_batch()

    def __iter__(self):
        return self

    def get_batch(self):
        xs, ys, xq, yq = [], [], [], []
        if self.source == 'train':
            x = self.unlabeled_x
            num_way = self.test_num_way

        elif self.source == 'val':
            x = self.val_x
            y = self.val_y
            class_list, _ = np.unique(y, return_counts=True)
            num_val_shot = 1

            num_way = 2

        for _ in range(self.tasks_per_batch):

            support_set = []
            query_set = []
            support_sety = []
            query_sety = []

            if self.source == 'val':

                classes = np.random.choice(class_list, num_way, replace=False)
                support_idx = []
                query_idx = []
                for k in classes:
                    k_idx = np.where(y == k)[0]
                    permutation = np.random.permutation(len(k_idx))
                    k_idx = k_idx[permutation]
                    support_idx.append(k_idx[:num_val_shot])
                    query_idx.append(k_idx[num_val_shot:num_val_shot + 30])
                support_idx = np.concatenate(support_idx)
                query_idx = np.concatenate(query_idx)

                support_x = x[support_idx]
                query_x = x[query_idx]
                s_y = y[support_idx]
                q_y = y[query_idx]
                support_y = copy.deepcopy(s_y)
                query_y = copy.deepcopy(q_y)

                i = 0
                for k in classes:
                    support_y[s_y == k] = i
                    query_y[q_y == k] = i
                    i += 1

                support_set.append(support_x)
                support_sety.append(support_y)
                query_set.append(query_x)
                query_sety.append(query_y)

            elif self.source == 'train':
                tmp_x = copy.deepcopy(x)
                min_count = 0
                while min_count < (self.shot + self.query):
                    # U(min_col, max_col)
                    min_col = int(x.shape[1] * 0.2)
                    max_col = int(x.shape[1] * 0.5)
                    # wybor kolumn
                    col = np.random.choice(range(min_col, max_col), 1, replace=False)[0]
                    task_idx = np.random.choice([i for i in range(x.shape[1])], col, replace=False)
                    # masked_x - wyfiltrowany podzbior kolumn
                    masked_x = np.ascontiguousarray(x[:, task_idx], dtype=np.float32)
                    # # k-means model
                    kmeans = faiss.Kmeans(masked_x.shape[1], num_way, niter=20, nredo=1, verbose=False,
                                          min_points_per_centroid=self.shot + self.query, gpu=1)
                    kmeans.train(masked_x)
                    # szukamy dla kazdego elementu z masked_x najblizszy cluster
                    # D - odleglosci do najblizszych klastrow
                    # I - indeksy najblizszych klastrow dla kazdego punktu
                    D, I = kmeans.index.search(masked_x, 1)
                    # y - indeksy klastrow -> integer label
                    y = I[:, 0].astype(np.int32)
                    # class_list - indeksy klastrów, counts - ile punktow w kazdym klastrze
                    class_list, counts = np.unique(y, return_counts=True)
                    min_count = min(counts)

                    # masked_val_x = np.ascontiguousarray(self.val_x[:, task_idx], dtype=np.float32)
                    # D_val, I_val = kmeans.index.search(masked_val_x, 1)
                    #
                    # from sklearn.metrics import adjusted_rand_score
                    # ari = adjusted_rand_score(self.val_y, I_val[:, 0])
                    # if ari < self.eps:
                    #     self.invalid_count += 1
                    #     print(f'Invalid count: {self.invalid_count}')
                    #     min_count = 0

                # num_to_permute - liczba wierszy w tabeli
                num_to_permute = x.shape[0]
                for t_idx in task_idx:
                    # permutacja wierszy z wyselekcjonowanych wczesniej kolumn
                    rand_perm = np.random.permutation(num_to_permute)
                    # w kazdej kolumnie permutujemy wartosci
                    tmp_x[:, t_idx] = tmp_x[:, t_idx][rand_perm]

                # wybor n_way klas z listy klas
                #print(class_list, num_way)
                #classes = class_list
                classes = np.random.choice(class_list, num_way, replace=False)

                # konstrukcja support i query setow
                support_idx = []
                query_idx = []
                for k in classes:
                    # indeksy danych z klasy k
                    k_idx = np.where(y == k)[0]
                    permutation = np.random.permutation(len(k_idx))
                    # shuffle, zeby podzielic na query i support
                    k_idx = k_idx[permutation]
                    support_idx.append(k_idx[:self.shot])
                    query_idx.append(k_idx[self.shot:self.shot + self.query])
                support_idx = np.concatenate(support_idx)
                query_idx = np.concatenate(query_idx)

                support_x = tmp_x[support_idx]
                query_x = tmp_x[query_idx]
                s_y = y[support_idx]
                q_y = y[query_idx]
                support_y = copy.deepcopy(s_y)
                query_y = copy.deepcopy(q_y)

                i = 0
                for k in classes:
                    support_y[s_y == k] = i
                    query_y[q_y == k] = i
                    i += 1

                support_set.append(support_x)
                support_sety.append(support_y)
                query_set.append(query_x)
                query_sety.append(query_y)

            xs_k = np.concatenate(support_set, 0)
            xq_k = np.concatenate(query_set, 0)
            ys_k = np.concatenate(support_sety, 0)
            yq_k = np.concatenate(query_sety, 0)

            xs.append(xs_k)
            xq.append(xq_k)
            ys.append(ys_k)
            yq.append(yq_k)

        xs, ys = np.stack(xs, 0), np.stack(ys, 0)
        xq, yq = np.stack(xq, 0), np.stack(yq, 0)

        if self.source == 'val':
            xs = np.reshape(
                xs,
                [self.tasks_per_batch, num_way * num_val_shot, self.tabular_size])
        else:
            xs = np.reshape(
                xs,
                [self.tasks_per_batch, num_way * self.shot, self.tabular_size])

        if self.source == 'val':
            xq = np.reshape(
                xq,
                [self.tasks_per_batch, num_way * 30, self.tabular_size])
        else:
            xq = np.reshape(
                xq,
                [self.tasks_per_batch, num_way * self.query, self.tabular_size])

        xs = xs.astype(np.float32)
        xq = xq.astype(np.float32)
        ys = ys.astype(np.float32)
        yq = yq.astype(np.float32)

        xs = torch.from_numpy(xs).type(torch.FloatTensor)
        xq = torch.from_numpy(xq).type(torch.FloatTensor)

        ys = torch.from_numpy(ys).type(torch.LongTensor)
        yq = torch.from_numpy(yq).type(torch.LongTensor)

        batch = {'train': [xs, ys], 'test': [xq, yq]}

        return batch

    def get_test_batch(self):
        num_classes = len(np.unique(self.test_y))
        tasks = []
        for _ in range(self.tasks_per_batch):
            support_set_x = []
            support_set_y = []
            query_set_x = []
            query_set_y = []

            selected_classes = np.random.choice(range(num_classes), self.num_classes, replace=False)
            for class_id in selected_classes:
                class_indices = np.where(self.test_y == class_id)[0]
                np.random.shuffle(class_indices)

                support_indices = class_indices[:self.shot]
                query_indices = class_indices[self.shot:self.shot + self.query]

                support_set_x.append(self.test_x[support_indices])
                support_set_y.append(np.full(len(support_indices), class_id))  # Class labels for support set
                query_set_x.append(self.test_x[query_indices])
                query_set_y.append(np.full(len(query_indices), class_id))  # Class labels for query set

            support_set_x = np.concatenate(support_set_x, axis=0)
            support_set_y = np.concatenate(support_set_y, axis=0)
            query_set_x = np.concatenate(query_set_x, axis=0)
            query_set_y = np.concatenate(query_set_y, axis=0)

            support_set_x = np.expand_dims(support_set_x, axis=0)  # Add task dimension
            support_set_y = np.expand_dims(support_set_y, axis=0)  # Add task dimension
            query_set_x = np.expand_dims(query_set_x, axis=0)  # Add task dimension
            query_set_y = np.expand_dims(query_set_y, axis=0)

            tasks.append({
                'train': [torch.tensor(support_set_x, dtype=torch.float32),
                          torch.tensor(support_set_y, dtype=torch.long)],
                'test': [torch.tensor(query_set_x, dtype=torch.float32), torch.tensor(query_set_y, dtype=torch.long)]
            })
        return tasks

    def get_invalid_count(self):
        return self.invalid_count
