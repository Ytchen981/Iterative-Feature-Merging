from collections import defaultdict
from re import L
from typing import NamedTuple
import time

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def axes2perm_to_perm2axes(axes_to_perm):
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return perm_to_axes

def self_weight_matching(perm_to_axes, params):
  perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
  distances = {}
  for p_name in perm_sizes.keys():
      n = perm_sizes[p_name]
      A = torch.zeros((n, n)).cuda()
      for wk, axis in perm_to_axes[p_name]:
          w_a = params[wk]
          #w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
          w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
          #w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

          A += w_a @ w_a.T
      distance_matrix = A + A.T - A.diag().unsqueeze(0) - A.diag().unsqueeze(1)
      distances[p_name] = distance_matrix.cpu()
  return distances

def self_weight_matching_p_name(perm_to_axes, params, p_name, n):
  A = torch.zeros((n, n)).cuda()
  for wk, axis in perm_to_axes[p_name]:
    w_a = params[wk]
    if len(w_a.shape) < 2 or "identity_transform" in wk:
        pass
    else:
        #w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
        w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
        # w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

        A += w_a @ w_a.T
  distance_matrix = A + A.T - A.diag().unsqueeze(0) - A.diag().unsqueeze(1)
  return distance_matrix.cpu()

def merge_channel_p_name(perm_to_axes, params, merge_num, p_name, n, i, j):
    min_ij = min(i, j)
    max_ij = max(i, j)
    indices = [num for num in range(min_ij)] + [num for num in range(min_ij + 1, max_ij)] + [num for num in range(max_ij + 1, n)] + [max_ij, min_ij]
    assert len(indices) == n
    merge_num_list = merge_num[p_name]
    merge_num_list = merge_num_list[indices]
    for wk, axis in perm_to_axes[p_name]:
        if wk == "conv1.weight" or wk == "conv1.bn" or "layer1" in wk:
            pass
        else:
            w_a = params[wk]
            assert axis in (0, 1)
            if axis == 0:
                w_a = w_a[indices]
                w_a[-2] += w_a[-1]
                params[wk] = w_a[:-1]
            else:
                w_a = w_a[:, indices]
                w_a[:, -2] = (w_a[:, -2] * merge_num_list[-2] + w_a[:, -1] * merge_num_list[-1]) / (
                            merge_num_list[-2] + merge_num_list[-1])
                params[wk] = w_a[:, :-1]
    merge_num_list[-2] += merge_num_list[-1]
    merge_num[p_name] =  merge_num_list[:-1]

    return params, merge_num

def self_merge_weight(perm_to_axes, params, max_ratio=0.5, threshold=0.1):
    perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    merge_num = {p: torch.ones(perm_sizes[p]) for p in perm_sizes.keys()}
    iter = 0
    time_used = []
    tick = time.time()
    while True:
        Flag = True
        for p_name in perm_sizes.keys():
            n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]
            if iter >= perm_sizes[p_name] * max_ratio:
                pass
            else:
                distance = self_weight_matching_p_name(perm_to_axes, params, p_name, n)
                value, indices = torch.topk(distance.view(-1), k=n + 1)
                v = value[-1]
                if v.item() < threshold * torch.min(distance).item():
                    pass
                else:
                    Flag = False
                    indices = indices[-1].item()
                    assert v <= 0
                    i = indices // n
                    j = int(indices % n)
                    assert distance[i, j].item() == v.item()
                    params, merge_num = merge_channel_p_name(perm_to_axes, params, merge_num, p_name, n, i, j)
        iter += 1
        '''tock = time.time()
        time_used.append(tock - tick)
        tick = tock'''
        #print(f"iter: {iter}, time_used: {time.time() - tick:.2f}s")
        if Flag:
            break
        '''if len(time_used) >=100:
            break
    print(f"mean:{np.mean(time_used)}; std:{np.std(time_used)}")'''
    new_perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    return params, new_perm_sizes

def self_merge_weight_get_dict(perm_to_axes, params, max_ratio=0.5, threshold=0.1):
    perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    merge_num = {p: torch.ones(perm_sizes[p]) for p in perm_sizes.keys()}
    distinct_features = {p: [[i] for i in range(perm_sizes[p])] for p in perm_sizes.keys()}
    iter = 0
    tick = time.time()
    while True:
        Flag = True
        for p_name in perm_sizes.keys():
            n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]
            if iter >= perm_sizes[p_name] * max_ratio:
                pass
            else:
                distance = self_weight_matching_p_name(perm_to_axes, params, p_name, n)
                value, indices = torch.topk(distance.view(-1), k=n + 1)
                v = value[-1]
                if v.item() < threshold * torch.min(distance).item():
                    pass
                else:
                    Flag = False
                    indices = indices[-1].item()
                    assert v <= 0
                    i = indices // n
                    j = int(indices % n)
                    assert distance[i, j].item() == v.item()
                    params, merge_num = merge_channel_p_name(perm_to_axes, params, merge_num, p_name, n, i, j)
                    min_ij = min(i, j)
                    max_ij = max(i, j)
                    list_i = distinct_features[p_name].pop(min_ij)
                    list_j = distinct_features[p_name].pop(max_ij-1)
                    distinct_features[p_name].append(list_i+list_j)
        iter += 1
        print(f"iter: {iter}, time_used: {time.time() - tick:.2f}s")
        if Flag:
            break
    new_perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    return params, new_perm_sizes, distinct_features