from re import I
import numpy as np
import torch
import os
import copy
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F


class Income(object):
    kmeans = None
    centroids = None
    def __init__(self, P, tabular_size, seed, source, shot, tasks_per_batch, test_num_way, query, eps):
        super().__init__()
        self.num_classes = 2
        self.tabular_size = tabular_size
        self.source = source
        self.shot = shot
        self.query = query
        self.tasks_per_batch = tasks_per_batch
        self.unlabeled_x = np.load('./data/income/train_x.npy')
        self.test_x = np.load('./data/income/xtest.npy')
        self.test_y = np.load('./data/income/ytest.npy')
        self.val_x = np.load('./data/income/val_x.npy')
        self.val_y = np.load(
            './data/income/pseudo_val_y.npy')
        self.test_num_way = test_num_way
        self.test_rng = np.random.RandomState(seed)
        self.val_rng = np.random.RandomState(seed)
        self.invalid_count = 0
        self.eps = eps

        if not Income.kmeans:
            Income.kmeans = faiss.Kmeans(self.unlabeled_x.shape[1], 150, niter=20, nredo=1, verbose=False, gpu=1)
            Income.kmeans.train(self.unlabeled_x)
            Income.centroids = Income.kmeans.centroids

        self.centroids = Income.centroids

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

                similarities = []
                for i, element in enumerate(support_x):
                    dot_products = np.dot(self.centroids, element)
                    norm_centroids = np.linalg.norm(self.centroids, axis=1)
                    norm_element = np.linalg.norm(element)
                    similarity = np.divide(dot_products, norm_element * norm_centroids,
                                           out=np.zeros_like(dot_products), where=(norm_element * norm_centroids) != 0)
                    similarities.append(similarity)

                similarities_query = []
                for i, element in enumerate(query_x):
                    dot_products = np.dot(self.centroids, element)
                    norm_centroids = np.linalg.norm(self.centroids, axis=1)
                    norm_element = np.linalg.norm(element)
                    similarity = np.divide(dot_products, norm_element * norm_centroids,
                                           out=np.zeros_like(dot_products), where=(norm_element * norm_centroids) != 0)
                    similarities_query.append(similarity)

                support_set.append(similarities)
                support_sety.append(support_y)
                query_set.append(similarities_query)
                query_sety.append(query_y)

            elif self.source == 'train':
                tmp_x = copy.deepcopy(x)
                min_count = 0
                while min_count < (self.shot + self.query):
                    min_col = int(x.shape[1] * 0.2)
                    max_col = int(x.shape[1] * 0.5)
                    col = np.random.choice(range(min_col, max_col), 1, replace=False)[0]
                    task_idx = np.random.choice([i for i in range(x.shape[1])], col, replace=False)
                    masked_x = np.ascontiguousarray(x[:, task_idx], dtype=np.float32)
                    kmeans = faiss.Kmeans(masked_x.shape[1], num_way, niter=20, nredo=1, verbose=False,
                                          min_points_per_centroid=self.shot + self.query, gpu=1)
                    kmeans.train(masked_x)
                    D, I = kmeans.index.search(masked_x, 1)
                    y = I[:, 0].astype(np.int32)
                    class_list, counts = np.unique(y, return_counts=True)
                    min_count = min(counts)

                    valid_classes = [cls for cls, count in zip(class_list, counts) if count >= (self.shot + self.query)]
                    if len(valid_classes) < num_way:
                        print("WARNING: Not enough valid clusters! Retrying...")
                        min_count = 0

                classes = class_list

                support_idx = []
                query_idx = []
                for k in classes:
                    k_idx = np.where(y == k)[0]
                    permutation = np.random.permutation(len(k_idx))
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

                remaining_idx = np.setdiff1d(np.arange(self.tabular_size), task_idx)

                similarities = []
                for i, element in enumerate(support_x):
                    element_filtered = element[remaining_idx]
                    centroids_filtered = self.centroids[:, remaining_idx]
                    dot_products = np.dot(centroids_filtered, element_filtered)
                    norm_centroids = np.linalg.norm(centroids_filtered, axis=1)
                    norm_element = np.linalg.norm(element_filtered)
                    similarity = np.divide(dot_products, norm_element * norm_centroids,
                                           out=np.zeros_like(dot_products), where=(norm_element * norm_centroids) != 0)
                    similarities.append(similarity)

                similarities_query = []
                for i, element in enumerate(query_x):
                    element_filtered = element[remaining_idx]
                    centroids_filtered = self.centroids[:, remaining_idx]
                    dot_products = np.dot(centroids_filtered, element_filtered)
                    norm_centroids = np.linalg.norm(centroids_filtered, axis=1)
                    norm_element = np.linalg.norm(element_filtered)
                    similarity = np.divide(dot_products, norm_element * norm_centroids,
                                           out=np.zeros_like(dot_products), where=(norm_element * norm_centroids) != 0)
                    similarities_query.append(similarity)

                support_set.append(similarities)
                support_sety.append(support_y)
                query_set.append(similarities_query)
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
                [self.tasks_per_batch, num_way * num_val_shot, self.centroids.shape[0]])
        else:
            xs = np.reshape(
                xs,
                [self.tasks_per_batch, num_way * self.shot, self.centroids.shape[0]])

        if self.source == 'val':
            xq = np.reshape(
                xq,
                [self.tasks_per_batch, num_way * 30, self.centroids.shape[0]])
        else:
            xq = np.reshape(
                xq,
                [self.tasks_per_batch, num_way * self.query, self.centroids.shape[0]])

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
        def transform(x):
            dot_products = np.dot(self.centroids, x)
            norm_centroids = np.linalg.norm(self.centroids, axis=1)
            norm_element = np.linalg.norm(x)
            similarity = np.divide(dot_products, norm_element * norm_centroids,
                                   out=np.zeros_like(dot_products), where=(norm_element * norm_centroids) != 0)
            return similarity

        num_classes = len(np.unique(self.test_y))
        tasks = []
        for _ in range(self.tasks_per_batch):
            support_set_x = []
            support_set_y = []
            query_set_x = []
            query_set_y = []

            selected_classes = np.random.choice(range(num_classes), self.test_num_way, replace=False)
            for class_id in selected_classes:
                class_indices = np.where(self.test_y == class_id)[0]
                np.random.shuffle(class_indices)

                support_indices = class_indices[:self.shot]
                query_indices = class_indices[self.shot:self.shot + self.query]

                current = self.test_x[support_indices]
                v = np.stack([transform(c) for c in current])
                support_set_x.append(v)

                # support_set_x.append(self.test_x[support_indices])
                support_set_y.append(np.full(len(support_indices), class_id))

                current_query = self.test_x[query_indices]
                v = np.stack([transform(c) for c in current_query])
                query_set_x.append(v)

                #query_set_x.append(self.test_x[query_indices])
                query_set_y.append(np.full(len(query_indices), class_id))  # Class labels for query set

            # Convert lists to proper arrays
            support_set_x = np.vstack(support_set_x)  # Shape: (num_shots * num_ways, 150)
            support_set_y = np.concatenate(support_set_y)  # Shape: (num_shots * num_ways,)

            query_set_x = np.vstack(query_set_x)  # Shape: (num_queries * num_ways, 150)
            query_set_y = np.concatenate(query_set_y)  # Shape: (num_queries * num_ways,)

            # Add batch dimension
            support_set_x = np.expand_dims(support_set_x, axis=0)  # Shape: (1, num_shots * num_ways, 150)
            query_set_x = np.expand_dims(query_set_x, axis=0)  # Shape: (1, num_queries * num_ways, 150)

            support_set_y = np.expand_dims(support_set_y, axis=0)  # Shape: (1, num_shots * num_ways)
            query_set_y = np.expand_dims(query_set_y, axis=0)  # Shape: (1, num_queries * num_ways)

            # Convert to PyTorch tensors
            tasks.append({
                'train': [torch.tensor(support_set_x, dtype=torch.float32),
                          torch.tensor(support_set_y, dtype=torch.long)],
                'test': [torch.tensor(query_set_x, dtype=torch.float32),
                         torch.tensor(query_set_y, dtype=torch.long)]
            })
        return tasks

    def get_invalid_count(self):
        return self.invalid_count
