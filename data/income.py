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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.unlabeled_x = torch.tensor(np.load('./data/income/train_x.npy'), dtype=torch.float32).to(self.device)
        self.test_x = torch.tensor(np.load('./data/income/xtest.npy'), dtype=torch.float32).to(self.device)
        self.test_y = torch.tensor(np.load('./data/income/ytest.npy'), dtype=torch.long).to(self.device)
        self.val_x = torch.tensor(np.load('./data/income/val_x.npy'), dtype=torch.float32).to(self.device)
        self.val_y = torch.tensor(np.load('./data/income/pseudo_val_y.npy'), dtype=torch.long).to(self.device)
        # self.unlabeled_x = np.load('./data/income/train_x.npy')
        # self.test_x = np.load('./data/income/xtest.npy')
        # self.test_y = np.load('./data/income/ytest.npy')
        # self.val_x = np.load('./data/income/val_x.npy')
        # self.val_y = np.load(
        #     './data/income/pseudo_val_y.npy')
        self.test_num_way = test_num_way
        self.test_rng = torch.Generator(device=self.device).manual_seed(seed)
        self.val_rng = torch.Generator(device=self.device).manual_seed(seed)
        self.invalid_count = 0
        self.eps = eps

        if not Income.kmeans:
            Income.kmeans = faiss.Kmeans(self.unlabeled_x.shape[1], P.kernel_size, niter=20, gpu=True)
            Income.kmeans.train(self.unlabeled_x.cpu().numpy())
            Income.centroids = torch.tensor(Income.kmeans.centroids, device=self.device)

        self.centroids = Income.centroids

    def __next__(self):
        return self.get_batch()

    def __iter__(self):
        return self

    def get_batch(self):
        xs, ys, xq, yq = [], [], [], []
        x = self.unlabeled_x if self.source == 'train' else self.val_x
        y = self.val_y if self.source == 'val' else None
        num_way = 2 if self.source == 'val' else self.test_num_way
        num_val_shot = 1 if self.source == 'val' else None
        # if self.source == 'train':
        #     x = self.unlabeled_x
        #     num_way = self.test_num_way
        #
        # elif self.source == 'val':
        #     x = self.val_x
        #     y = self.val_y
        #     class_list, _ = np.unique(y, return_counts=True)
        #     num_val_shot = 1
        #
        #     num_way = 2

        for _ in range(self.tasks_per_batch):

            support_set = []
            query_set = []
            support_sety = []
            query_sety = []

            if self.source == 'val':
                class_list = torch.unique(y)
                classes = class_list[torch.randperm(len(class_list))[:num_way]]
                support_idx = []
                query_idx = []

                for k in classes:
                    k_idx = (y == k).nonzero(as_tuple=True)[0]
                    k_idx = k_idx[torch.randperm(len(k_idx))]
                    support_idx.append(k_idx[:num_val_shot])
                    query_idx.append(k_idx[num_val_shot:num_val_shot + 30])
                support_idx, query_idx = torch.cat(support_idx), torch.cat(query_idx)

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

                support_set.append(self.compute_similarity(support_x))
                query_set.append(self.compute_similarity(query_x))
                support_sety.append(support_y)
                query_sety.append(query_y)

            elif self.source == 'train':
                tmp_x = copy.deepcopy(x)
                min_count = 0
                while min_count < (self.shot + self.query):
                    min_col = int(x.shape[1] * 0.2)
                    max_col = int(x.shape[1] * 0.5)
                    col = torch.randint(min_col, max_col, (1,)).item()
                    task_idx = torch.randperm(x.shape[1], device=self.device)[:col]
                    masked_x = x[:, task_idx].contiguous()
                    kmeans = faiss.Kmeans(masked_x.shape[1], num_way, niter=20, nredo=1, verbose=False,
                                          min_points_per_centroid=self.shot + self.query, gpu=1)
                    kmeans.train(masked_x.cpu().numpy())
                    D, I = kmeans.index.search(masked_x.cpu().numpy(), 1)
                    y = torch.tensor(I[:, 0], device=self.device, dtype=torch.int32)
                    class_list, counts = torch.unique(y, return_counts=True)
                    min_count = counts.min().item()

                    valid_classes = class_list[(counts >= (self.shot + self.query))]
                    if valid_classes.numel() < num_way:
                        print("WARNING: Not enough valid clusters! Retrying...")
                        min_count = 0

                classes = valid_classes.tolist()

                support_idx = []
                query_idx = []
                for k in classes:
                    k_idx = (y == k).nonzero(as_tuple=True)[0]
                    perm = torch.randperm(k_idx.size(0), device=self.device)
                    k_idx = k_idx[perm]
                    support_idx.append(k_idx[:self.shot])
                    query_idx.append(k_idx[self.shot:self.shot + self.query])
                support_idx = torch.cat(support_idx)
                query_idx = torch.cat(query_idx)

                support_x = tmp_x[support_idx]
                query_x = tmp_x[query_idx]
                s_y = y[support_idx]
                q_y = y[query_idx]
                support_y, query_y = s_y.clone(), q_y.clone()

                i = 0
                for k in classes:
                    support_y[s_y == k] = i
                    query_y[q_y == k] = i
                    i += 1

                remaining_idx = torch.tensor(list(set(range(self.tabular_size)) - set(task_idx.tolist())), device=self.device)

                def compute_similarities(data_x):
                    similarities = []
                    centroids_filtered = self.centroids[:, remaining_idx]
                    for element in data_x:
                        element_filtered = element[remaining_idx]
                        dot_products = torch.matmul(centroids_filtered, element_filtered)
                        norm_centroids = torch.norm(centroids_filtered, dim=1)
                        norm_element = torch.norm(element_filtered)
                        similarity = torch.div(dot_products, norm_element * norm_centroids)
                        similarity = torch.where((norm_element * norm_centroids) != 0, similarity,
                                                 torch.zeros_like(dot_products))
                        similarities.append(similarity)
                    return similarities

                support_set.append(compute_similarities(support_x))
                support_sety.append(support_y)
                query_set.append(compute_similarities(query_x))
                query_sety.append(query_y)

            xs_k = torch.cat([item for sublist in support_set for item in sublist], dim=0)
            xq_k = torch.cat([item for sublist in query_set for item in sublist], dim=0)
            ys_k = torch.cat(
                [item.unsqueeze(0) if item.dim() == 0 else item for sublist in support_sety for item in sublist], dim=0)
            yq_k = torch.cat(
                [item.unsqueeze(0) if item.dim() == 0 else item for sublist in query_sety for item in sublist], dim=0)

            xs.append(xs_k)
            xq.append(xq_k)
            ys.append(ys_k)
            yq.append(yq_k)
        xs, ys = torch.stack(xs, 0), torch.stack(ys, 0)
        xq, yq = torch.stack(xq, 0), torch.stack(yq, 0)

        if self.source == 'val':
            xs = torch.reshape(
                xs,
                [self.tasks_per_batch, num_way * num_val_shot, self.centroids.shape[0]])
        else:
            xs = torch.reshape(
                xs,
                [self.tasks_per_batch, num_way * self.shot, self.centroids.shape[0]])
        if self.source == 'val':
            xq = torch.reshape(
                xq,
                [self.tasks_per_batch, num_way * 30, self.centroids.shape[0]])
        else:
            xq = torch.reshape(
                xq,
                [self.tasks_per_batch, num_way * self.query, self.centroids.shape[0]])

        xs = torch.tensor(xs, dtype=torch.float32, device=self.device)
        xq = torch.tensor(xq, dtype=torch.float32, device=self.device)
        ys = torch.tensor(ys, dtype=torch.float32, device=self.device)
        yq = torch.tensor(yq, dtype=torch.float32, device=self.device)

        batch = {'train': [xs, ys], 'test': [xq, yq]}

        return batch

    def get_test_batch(self):
        def transform(x):
            dot_products = torch.matmul(self.centroids, x)
            norm_centroids = torch.norm(self.centroids, dim=1)
            norm_element = torch.norm(x)
            similarity = torch.div(dot_products, norm_element * norm_centroids)
            similarity = torch.where((norm_element * norm_centroids) != 0, similarity,
                                     torch.zeros_like(dot_products))
            return similarity

        num_classes = len(torch.unique(self.test_y))
        tasks = []
        for _ in range(self.tasks_per_batch):
            support_set_x = []
            support_set_y = []
            query_set_x = []
            query_set_y = []

            selected_classes = torch.randperm(num_classes, device=self.device)[:self.test_num_way]
            for class_id in selected_classes:
                class_indices = (self.test_y == class_id).nonzero(as_tuple=True)[0]
                shuffled_indices = class_indices[torch.randperm(class_indices.size(0), device=self.device)]

                support_indices = shuffled_indices[:self.shot]
                query_indices = shuffled_indices[self.shot:self.shot + self.query]

                current = self.test_x[support_indices]
                v = torch.stack([transform(c) for c in current])
                support_set_x.append(v)

                # support_set_x.append(self.test_x[support_indices])
                support_set_y.append(torch.full((len(support_indices),), class_id, dtype=torch.long, device=self.device))

                current_query = self.test_x[query_indices]
                v = torch.stack([transform(c) for c in current_query])
                query_set_x.append(v)

                #query_set_x.append(self.test_x[query_indices])
                query_set_y.append(torch.full((len(query_indices),), class_id, dtype=torch.long, device=self.device))

            support_set_x = torch.cat(support_set_x).unsqueeze(0)  # Add batch dimension
            support_set_y = torch.cat(support_set_y).unsqueeze(0)

            query_set_x = torch.cat(query_set_x).unsqueeze(0)
            query_set_y = torch.cat(query_set_y).unsqueeze(0)

            tasks.append({
                'train': [support_set_x.to(self.device), support_set_y.to(self.device)],
                'test': [query_set_x.to(self.device), query_set_y.to(self.device)]
            })
        return tasks

    def compute_similarity(self, elements):
        dot_products = torch.mm(self.centroids, elements.T).T
        norm_centroids = torch.norm(self.centroids, dim=1, keepdim=True)
        norm_elements = torch.norm(elements, dim=1, keepdim=True)
        return dot_products / (norm_elements * norm_centroids.T).clamp(min=1e-8)
