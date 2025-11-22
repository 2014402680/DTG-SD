import logging
import datetime
import math
import sys
import gc
import json
import time
import torch.nn.functional as F
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import re
import os
import sys
import json
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random
from sklearn.metrics import precision_recall_fscore_support
import functools
import dgl

try:
    from torch.utils.tensorboard import SummaryWriter
except BaseException as e:
    from tensorboardX import SummaryWriter

from dataset import Sampler, EdgeSeqDataset, GraphAdjDataset, GraphAdjDataset_DGL_Input
from utils import anneal_fn, get_enc_len, pretrain_load_data, get_linear_schedule_with_warmup, \
    bp_compute_abmae, compareloss, label2onehot, correctness_GPU, correctness, macrof1, weightf1, \
    few_shot_split_graphlevel, \
    distance2center, center_embedding, index2mask, mask_select_emb
from gin import GIN
from node_prompt_layer import node_prompt_layer_linear_mean, node_prompt_layer_linear_sum, \
    node_prompt_layer_feature_weighted_mean, node_prompt_layer_feature_weighted_sum, node_prompt_layer_sum

warnings.filterwarnings("ignore")
INF = float("inf")
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from utils import map_activation_str_to_layer, split_and_batchify_graph_feats, GetAdj



class SupervisedTeacherGIN(torch.nn.Module):
    def __init__(self, config):
        super(SupervisedTeacherGIN, self).__init__()

        self.act = torch.nn.ReLU()
        self.g_net, self.bns, g_dim = self.create_net(
            name="graph", input_dim=config["node_feature_dim"], hidden_dim=config["gcn_hidden_dim"],
            num_layers=config["gcn_graph_num_layers"], num_bases=config["gcn_num_bases"],
            regularizer=config["gcn_regularizer"])
        self.num_layers_num = config["gcn_graph_num_layers"]
        self.dropout = torch.nn.Dropout(p=config["dropout"])

    def create_net(self, name, input_dim, **kw):
        num_layers = kw.get("num_layers", 1)
        hidden_dim = kw.get("hidden_dim", 64)
        num_rels = kw.get("num_rels", 1)
        num_bases = kw.get("num_bases", 8)
        regularizer = kw.get("regularizer", "basis")
        dropout = kw.get("dropout", 0.5)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):
            if i:
                nn = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), self.act,
                                         torch.nn.Linear(hidden_dim, hidden_dim))
            else:
                nn = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), self.act,
                                         torch.nn.Linear(hidden_dim, hidden_dim))
            conv = dgl.nn.pytorch.conv.GINConv(apply_func=nn, aggregator_type='sum')
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns, hidden_dim

    def forward(self, graph, graph_len, graphtask=False):
        graph_output = graph.ndata["feature"]
        xs = []
        for i in range(self.num_layers_num):
            graph_output = F.relu(self.convs[i](graph, graph_output))
            graph_output = self.bns[i](graph_output)
            graph_output = self.dropout(graph_output)
            xs.append(graph_output)
        xpool = []
        for x in xs:
            if graphtask:
                graph_embedding = split_and_batchify_graph_feats(x, graph_len)[0]
            else:
                graph_embedding = x
            graph_embedding = torch.sum(graph_embedding, dim=1)
            xpool.append(graph_embedding)
        x = torch.cat(xpool, -1)
        # x is graph level embedding; xs is node level embedding
        return x, torch.cat(xs, -1)


train_config = {
    "max_npv": 8,
    "max_npe": 8,
    "max_npvl": 8,
    "max_npel": 8,
    "max_ngv": 126,
    "max_nge": 282,
    "max_ngvl": 3,
    "max_ngel": 2,
    "base": 2,
    "gpu_id": 0,
    "num_workers": 12,
    "epochs": 100,
    "batch_size": 1024,
    "update_every": 1,
    "print_every": 100,
    "init_emb": "Equivariant",
    "share_emb": False,
    "share_arch": False,
    "dropatt": 0.2,
    "reg_loss": "MAE",
    "bp_loss": "CROSS",
    "bp_loss_slp": "anneal_cosine$1.0$0.01",
    "lr": 0.0001,
    "dropout": 0.03,
    "weight_decay": 0.0001,
    "max_grad_norm": 8,
    "pretrain_model": "GIN",
    "emb_dim": 128,
    "activation_function": "leaky_relu",
    "filter_net": "MaxGatedFilterNet",
    "predict_net": "SumPredictNet",
    "predict_net_add_enc": True,
    "predict_net_add_degree": True,
    "txl_graph_num_layers": 3,
    "txl_pattern_num_layers": 3,
    "txl_d_model": 128,
    "txl_d_inner": 128,
    "txl_n_head": 4,
    "txl_d_head": 4,
    "txl_pre_lnorm": True,
    "txl_tgt_len": 64,
    "txl_ext_len": 0,
    "txl_mem_len": 64,
    "txl_clamp_len": -1,
    "txl_attn_type": 0,
    "txl_same_len": False,
    "gcn_num_bases": 1,
    "gcn_regularizer": "bdd",
    "gcn_graph_num_layers": 3,
    "gcn_hidden_dim": 32,
    "gcn_ignore_norm": False,
    "save_pretrain_model_dir": "D:/DTI/R-HGNN-master/dumps/debug",
    "graphslabel_dir": "../data/debug/graphs",
    "downstream_graph_dir": "../data/debug/graphs",
    "downstream_save_data_dir": "../data/debug",
    "downstream_save_model_dir": "../dumps/ENZYMESNodeClassification/Prompt/GIN-FEATURE-WEIGHTED-SUM/all/1train1val10task",
    "downstream_graphslabel_dir": "../data/debug/graphs",
    "temperature": 0.01,
    "graph_finetuning_input_dim": 8,
    "graph_finetuning_output_dim": 2,
    "graph_label_num": 6,
    "seed": 0,
    "update_pretrain": False,
    "gcn_output_dim": 8,
    "prompt": "FEATURE-WEIGHTED-SUM",  # FEATURE-WEIGHTED-SUM
    "prompt_output_dim": 2,
    "scalar": 1e3,
    "dataset_seed": 0,
    "train_shotnum": 1,
    "val_shotnum": 1,
    "few_shot_tasknum": 10,
    "save_fewshot_dir": "D:/DTI/R-HGNN-master/data/ENZYMES/nodefewshot",
    "downstream_dropout": 0,
    "node_feature_dim": 256,
    "train_label_num": 2,
    "val_label_num": 2,
    "test_label_num": 2,
    "nhop_neighbour": 1,
    "graph_num": 1,
    "split_drop": False,
    "process_raw": False,
    "split": False,
    "use_supervised_teacher": True,
    "teacher_epochs": 300,
    "teacher_lr": 0.001,
    "teacher_weight_decay": 0.0001,
    "knowledge_distillation_weight": 0.5,
    "temperature_kd": 3.0,

    "data_percentage": "5%"

}


def pre_train(model, graph, device, config):
    epoch_step = 1
    total_step = config["epochs"] * epoch_step
    total_reg_loss = 0
    total_bp_loss = 0

    if config["update_pretrain"]:
        model.train()
    else:
        model.eval()

    total_time = 0
    graph = graph.to(device)
    graph_len = torch.tensor(graph.number_of_nodes(), device=device)
    s = time.time()

    x, pred = model(graph, graph_len, False)
    pred = F.sigmoid(pred)


    adj = graph.adjacency_matrix()
    adj = adj.to(device)

    if config["nhop_neighbour"] == 0:
        pred = pred
    else:
        for i in range(config["nhop_neighbour"]):

            pred = adj @ pred


    return pred


def train_supervised_teacher(teacher_model, graph, node_labels, train_mask, val_mask, device, config, logger=None):

    teacher_model.train()


    teacher_optimizer = torch.optim.AdamW(
        teacher_model.parameters(),
        lr=config["teacher_lr"],
        weight_decay=config["teacher_weight_decay"]
    )


    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    best_teacher_embedding = None

    graph = graph.to(device)
    graph_len = torch.tensor(graph.number_of_nodes(), device=device)

    for epoch in range(config["teacher_epochs"]):
        teacher_model.train()
        teacher_optimizer.zero_grad()

        _, teacher_embedding = teacher_model(graph, graph_len, False)


        train_logits = teacher_embedding[train_mask]
        train_labels = node_labels[train_mask]
        train_loss = criterion(train_logits, train_labels)


        train_loss.backward()
        teacher_optimizer.step()


        if epoch % 1 == 0:
            teacher_model.eval()
            with torch.no_grad():
                _, val_embedding = teacher_model(graph, graph_len, False)
                val_logits = val_embedding[val_mask]
                val_labels = node_labels[val_mask]
                val_pred = torch.argmax(val_logits, dim=1)
                val_acc = (val_pred == val_labels).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_teacher_embedding = teacher_embedding.detach().clone()

                # print(f"Teacher Epoch {epoch}: Train Loss {train_loss.item():.4f}, Val Acc {val_acc:.4f}")
                # if logger:
                #     logger.info(f"Teacher Epoch {epoch}: Train Loss {train_loss.item():.4f}, Val Acc {val_acc:.4f}")

    return best_teacher_embedding


def train_with_knowledge_distillation(model, teacher_embedding, optimizer, scheduler, data_type, device, config, epoch,
                                      label_num, pretrain_embedding, node_label, logger=None, writer=None):
    epoch_step = 1
    total_step = config["epochs"] * epoch_step
    total_reg_loss = 0
    total_bp_loss = 0
    batchcnt = 0
    total_acc = 0
    total_cnt = 1e-6

    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target)
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "NLL":
        reg_crit = lambda pred, target: F.nll_loss(F.relu(pred), target)
    elif config["reg_loss"] == "ABMAE":
        reg_crit = lambda pred, target: bp_compute_abmae(F.leaky_relu(pred), target) + 0.8 * F.l1_loss(F.relu(pred),
                                                                                                       target)
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "NLL":
        bp_crit = lambda pred, target, neg_slp: F.nll_loss(pred, target)
    elif config["bp_loss"] == "CROSS":
        bp_crit = lambda pred, target, neg_slp: F.cross_entropy(pred, target)
    elif config["bp_loss"] == "ABMAE":
        bp_crit = lambda pred, target, neg_slp: bp_compute_abmae(F.leaky_relu(pred, neg_slp), target) + 0.8 * F.l1_loss(
            F.leaky_relu(pred, neg_slp), target, reduce="none")
    else:
        raise NotImplementedError

    kd_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

    model.train()

    total_time = 0
    label_num = torch.tensor(label_num).to(device)
    batchcnt += 1
    s = time.time()

    embedding = model(pretrain_embedding, 0)


    node_label = node_label
    c_embedding = center_embedding(embedding, node_label, label_num)
    distance = distance2center(teacher_embedding, c_embedding) + 1e-6

    distance = 1 / F.normalize(distance, dim=1)
    pred = F.log_softmax(distance, dim=1)


    reg_loss = reg_crit(pred, node_label.type(torch.LongTensor).to(device))
    reg_loss.requires_grad_(True)

    distance2 = distance2center(embedding, c_embedding) + 1e-6
    distance2 = 1 / F.normalize(distance2, dim=1)
    pred2 = F.log_softmax(distance2, dim=1)

    pred_class = torch.argmax(pred, dim=1, keepdim=True)
    accuracy = correctness_GPU(pred_class, node_label)
    total_acc += accuracy

    if isinstance(config["bp_loss_slp"], (int, float)):
        neg_slp = float(config["bp_loss_slp"])
    else:
        bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
        neg_slp = anneal_fn(bp_loss_slp, 0 + epoch * epoch_step, T=total_step // 4, lambda0=float(l0),
                            lambda1=float(l1))

    bp_loss = bp_crit(pred2.float(), node_label.squeeze().type(torch.LongTensor).to(device), neg_slp)
    bp_loss.requires_grad_(True)


    if teacher_embedding is not None and config["use_supervised_teacher"]:

        teacher_logits = F.log_softmax(teacher_embedding / config["temperature_kd"], dim=1)
        student_logits = F.log_softmax(embedding / config["temperature_kd"], dim=1)


        kd_loss = kd_loss_fn(student_logits, F.softmax(teacher_logits, dim=1)) * (config["temperature_kd"] ** 2)


        total_loss = bp_loss + config["knowledge_distillation_weight"] * kd_loss + reg_loss
    else:
        kd_loss = torch.tensor(0.0)
        total_loss = bp_loss + reg_loss


    reg_loss_item = reg_loss.item()
    bp_loss_item = bp_loss.item()
    kd_loss_item = kd_loss.item() if isinstance(kd_loss, torch.Tensor) else 0.0
    total_reg_loss += reg_loss_item
    total_bp_loss += bp_loss_item

    if writer:
        writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item, epoch * epoch_step + 0)
        writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item, epoch * epoch_step + 0)
        writer.add_scalar("%s/KD-Loss" % data_type, kd_loss_item, epoch * epoch_step + 0)

    if logger and (0 % config["print_every"] == 0 or 0 == epoch_step - 1):
        logger.info(
            "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:0>5.8f}\tbp loss: {:0>5.8f}\tkd loss: {:0>5.8f}".format(
                epoch, config["epochs"], data_type, 0, epoch_step, reg_loss_item, bp_loss_item, kd_loss_item))


    if (config["update_every"] < 2 or 0 % config["update_every"] == 0 or 0 == epoch_step - 1):
        if config["max_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        if scheduler is not None:
            scheduler.step(epoch * epoch_step + 0)

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

    e = time.time()
    total_time += e - s

    if writer:
        writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), reg_loss.item(), epoch)
        writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), bp_loss.item(), epoch)
        writer.add_scalar("%s/KD-Loss-epoch" % data_type, kd_loss_item, epoch)

    if logger:
        logger.info(
            "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>5.8f}\tbp loss: {:0>5.8f}\tkd loss: {:0>5.8f}\tmean_acc: {:0>1.3f}".format(
                epoch, config["epochs"], data_type, reg_loss.item(), bp_loss.item(), kd_loss_item, accuracy))

    gc.collect()
    return reg_loss, bp_loss, total_time, accuracy, c_embedding


def train(model, optimizer, scheduler, data_type, device, config, epoch, label_num, pretrain_embedding, node_label,
          logger=None, writer=None, teacher_embedding=None):

    if config["use_supervised_teacher"] and teacher_embedding is not None:
        return train_with_knowledge_distillation(model, teacher_embedding, optimizer, scheduler, data_type, device,
                                                 config, epoch, label_num, pretrain_embedding, node_label, logger,
                                                 writer)
    else:
        return train_2(model, optimizer, scheduler, data_type, device, config, epoch, label_num,
                              pretrain_embedding, node_label, logger, writer)


def train_2(model, optimizer, scheduler, data_type, device, config, epoch, label_num, pretrain_embedding,
                   node_label, logger=None, writer=None):
    epoch_step = 1
    total_step = config["epochs"] * epoch_step
    total_reg_loss = 0
    total_bp_loss = 0
    batchcnt = 0
    total_acc = 0
    total_cnt = 1e-6

    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target)
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "NLL":
        reg_crit = lambda pred, target: F.nll_loss(F.relu(pred), target)
    elif config["reg_loss"] == "ABMAE":
        reg_crit = lambda pred, target: bp_compute_abmae(F.leaky_relu(pred), target) + 0.8 * F.l1_loss(F.relu(pred),
                                                                                                       target)
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "NLL":
        bp_crit = lambda pred, target, neg_slp: F.nll_loss(pred, target)
    elif config["bp_loss"] == "CROSS":
        bp_crit = lambda pred, target, neg_slp: F.cross_entropy(pred, target)
    elif config["bp_loss"] == "ABMAE":
        bp_crit = lambda pred, target, neg_slp: bp_compute_abmae(F.leaky_relu(pred, neg_slp), target) + 0.8 * F.l1_loss(
            F.leaky_relu(pred, neg_slp), target, reduce="none")
    else:
        raise NotImplementedError

    model.train()

    total_time = 0
    label_num = torch.tensor(label_num).to(device)
    batchcnt += 1
    s = time.time()

    embedding = model(pretrain_embedding, 0)


    node_label = node_label
    c_embedding = center_embedding(embedding, node_label, label_num)
    distance = distance2center(embedding, c_embedding) + 1e-6

    distance = 1 / F.normalize(distance, dim=1)
    pred = F.log_softmax(distance, dim=1)
    reg_loss = reg_crit(pred, node_label.type(torch.LongTensor).to(device))

    reg_loss.requires_grad_(True)
    pred_class = torch.argmax(pred, dim=1, keepdim=True)
    accuracy = correctness_GPU(pred_class, node_label)
    total_acc += accuracy

    if isinstance(config["bp_loss_slp"], (int, float)):
        neg_slp = float(config["bp_loss_slp"])
    else:
        bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
        neg_slp = anneal_fn(bp_loss_slp, 0 + epoch * epoch_step, T=total_step // 4, lambda0=float(l0),
                            lambda1=float(l1))

    bp_loss = bp_crit(pred.float(), node_label.squeeze().type(torch.LongTensor).to(device), neg_slp)
    bp_loss.requires_grad_(True)

    reg_loss_item = reg_loss.item()
    bp_loss_item = bp_loss.item()
    total_reg_loss += reg_loss_item
    total_bp_loss += bp_loss_item

    if writer:
        writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item, epoch * epoch_step + 0)
        writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item, epoch * epoch_step + 0)

    if logger and (0 % config["print_every"] == 0 or 0 == epoch_step - 1):
        logger.info(
            "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:0>5.8f}\tbp loss: {:0>5.8f}".format(
                epoch, config["epochs"], data_type, 0, epoch_step, reg_loss_item, bp_loss_item))

    if (config["update_every"] < 2 or 0 % config["update_every"] == 0 or 0 == epoch_step - 1):
        if config["max_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        if scheduler is not None:
            scheduler.step(epoch * epoch_step + 0)

        optimizer.zero_grad()
        bp_loss.backward(retain_graph=True)
        reg_loss.backward(retain_graph=True)
        optimizer.step()

    e = time.time()
    total_time += e - s

    if writer:
        writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), reg_loss.item(), epoch)
        writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), bp_loss.item(), epoch)
    if logger:
        logger.info(
            "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>5.8f}\tbp loss: {:0>5.8f}\tmean_acc: {:0>1.3f}".format(
                epoch, config["epochs"], data_type, reg_loss.item(), bp_loss.item(), accuracy))
    gc.collect()
    return reg_loss, bp_loss, total_time, accuracy, c_embedding


def evaluate(model, data_type, device, config, epoch, c_embedding, label_num, pretrain_embedding, node_label, count,
             task, debug=False, logger=None, writer=None):
    epoch_step = 1
    total_reg_loss = 0
    total_step = config["epochs"] * epoch_step
    total_bp_loss = 0
    batchcnt = 0
    total_acc = 0
    total_macrof = 0
    total_weighted = 0
    total_cnt = 1e-6

    evaluate_results = {"data": {"id": list(), "counts": list(), "pred": list()},
                        "error": {"mae": INF, "mse": INF},
                        "time": {"avg": list(), "total": 0.0}}

    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "NLL":
        reg_crit = lambda pred, target: F.nll_loss(F.relu(pred), target)
    elif config["reg_loss"] == "ABMAE":
        reg_crit = lambda pred, target: bp_compute_abmae(F.relu(pred), target) + 0.8 * F.l1_loss(F.relu(pred), target,
                                                                                                 reduce="none")
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "NLL":
        bp_crit = lambda pred, target, neg_slp: F.nll_loss(pred, target)
    elif config["bp_loss"] == "CROSS":
        bp_crit = lambda pred, target, neg_slp: F.cross_entropy(pred, target)
    elif config["bp_loss"] == "ABMAE":
        bp_crit = lambda pred, target, neg_slp: bp_compute_abmae(F.leaky_relu(pred, neg_slp), target) + 0.8 * F.l1_loss(
            F.leaky_relu(pred, neg_slp), target, reduce="none")
    else:
        raise NotImplementedError

    model.eval()
    l2onehot = label2onehot(train_config["graph_label_num"], device)
    label_num = torch.tensor(label_num).to(device)
    total_time = 0
    batchcnt += 1

    s = time.time()

    embedding = model(pretrain_embedding, 0) * train_config["scalar"]

    c_embedding = center_embedding(embedding, node_label, label_num, debug)

    distance = distance2center(embedding, c_embedding)
    distance = -1 * F.normalize(distance, dim=1)

    pred = F.log_softmax(distance, dim=1)
    reg_loss = reg_crit(pred, node_label.type(torch.LongTensor).to(device))

    if isinstance(config["bp_loss_slp"], (int, float)):
        neg_slp = float(config["bp_loss_slp"])
    else:
        bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
        neg_slp = 0.2
    bp_loss = bp_crit(pred, node_label.squeeze().type(torch.LongTensor).to(device), neg_slp)

    pred_class = torch.argmax(pred, dim=1, keepdim=True)
    accuracy = correctness_GPU(pred_class, node_label)
    eval_pred = pred_class.cpu().numpy()
    eval_graph_label = node_label.cpu().numpy()
    acc = correctness(eval_pred, eval_graph_label)
    macrof = macrof1(eval_pred, eval_graph_label)
    weightf = weightf1(eval_pred, eval_graph_label)
    total_acc += acc
    total_macrof += macrof
    total_weighted += weightf

    reg_loss_item = reg_loss.item()
    bp_loss_item = bp_loss.item()

    if writer:
        writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item, epoch * epoch_step + 0)
        writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item, epoch * epoch_step + 0)

    if logger and 0 == epoch_step - 1:
        logger.info(
            "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:0>5.8f}\tbp loss: {:0>5.8f}\taccuracy: {:0>1.3f}".format(
                epoch, config["epochs"], data_type, 0, epoch_step, reg_loss_item, bp_loss_item, accuracy))

    if writer:
        writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), reg_loss.item(), epoch)
        writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), bp_loss.item(), epoch)

    if logger:
        logger.info(
            "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>5.8f}\tbp loss: {:0>5.8f}\tacc:{:0>1.3f}".format(
                epoch, config["epochs"], data_type, reg_loss_item, bp_loss_item, acc))

    gc.collect()
    return reg_loss_item, bp_loss_item, evaluate_results, total_time, acc, macrof, weightf, c_embedding, accuracy


if __name__ == "__main__":
    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        value = sys.argv[i + 1]

        if arg.startswith("--"):
            arg = arg[2:]
        if arg not in train_config:
            print("Warning: %s is not surported now." % (arg))
            continue
        train_config[arg] = value
        try:
            value = eval(value)
            if isinstance(value, (int, float)):
                train_config[arg] = value
        except:
            pass
    torch.set_printoptions(precision=10)

    torch.manual_seed(train_config["seed"])
    np.random.seed(train_config["seed"])

    root_fewshot_dir = os.path.join(train_config["save_fewshot_dir"], "%s_trainshot_%s_valshot_%s_tasks" %
                                    (train_config["train_shotnum"], train_config["val_shotnum"],
                                     train_config["few_shot_tasknum"]))

    dataset = 'luo'
    if dataset == 'luo':
        flow_task = ['drug_chemical', 'drug_sideeffects', 'drug_substituent', 'drug_target', 'target_go']
        for task in flow_task:
            none_all_acc = []
            acc = list()
            macroF = list()
            weightedF = list()

            for num in range(train_config["graph_num"]):
                ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                pretrain_model_name = "%s_%s_%s" % (train_config["pretrain_model"], train_config["predict_net"], ts)
                save_model_dir = train_config["downstream_save_model_dir"]
                save_pretrain_model_dir = "./save_model/zheng_model/10000_max_test/0.008_0.1"



                os.makedirs(save_model_dir, exist_ok=True)


                with open(os.path.join(save_model_dir, "train_config.json"), "w") as f:
                    json.dump(train_config, f)


                logger = logging.getLogger()
                logger.setLevel(logging.INFO)
                fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
                # console = logging.StreamHandler()
                # console.setFormatter(fmt)
                # logger.addHandler(console)
                logfile = logging.FileHandler(os.path.join(save_model_dir, "train_log.txt"), 'w')
                logfile.setFormatter(fmt)
                logger.addHandler(logfile)

                device = torch.device("cuda:%d" % train_config["gpu_id"] if train_config["gpu_id"] != -1 else "cpu")
                if train_config["gpu_id"] != -1:
                    torch.cuda.set_device(device)

                if train_config["share_emb"]:
                    train_config["max_npv"], train_config["max_npvl"], train_config["max_npe"], train_config[
                        "max_npel"] = \
                        train_config["max_ngv"], train_config["max_ngvl"], train_config["max_nge"], train_config[
                            "max_ngel"]

                fewshot_dir = os.path.join(root_fewshot_dir, str(num))
                print(os.path.exists(fewshot_dir))
                print("Load Few Shot")

                trainset = []
                valset = []
                testset = []

                # # 加载数据
                mid = np.load('node_data/zheng/10000_node/1-shot/{}/{}_train.npy'.format(task, task), allow_pickle=True)
                for item in mid:
                    trainset.append(list(item))

                mid = np.load('node_data/zheng/10000_node/1-shot/{}/{}_val.npy'.format(task, task), allow_pickle=True)
                for item in mid:
                    valset.append(list(item))

                mid = np.load('node_data/zheng/10000_node/1-shot/{}/{}_test.npy'.format(task, task), allow_pickle=True)
                for item in mid:
                    testset.append(list(item))


                trainset = torch.tensor(trainset, dtype=int)
                valset = torch.tensor(valset, dtype=int)
                testset = torch.tensor(testset, dtype=int)


                graph = \
                dgl.load_graphs("../get_nodefeature_module/graph_dti/zheng_graph_dti/n(6,3)_all_link_random/graph_all")[
                    0][num]


                graph.ndata["feature"] = graph.ndata["feature"][:, :256]
                nodelabel = graph.ndata["label"]
                num_node = len(nodelabel)
                nodelabel = torch.tensor([i[0] for i in nodelabel][:num_node])
                nodenum = num_node
                # nodenum=120000
                nodelabelnum = 2

                for count in range(train_config["few_shot_tasknum"]):
                    # 1
                    if train_config["pretrain_model"] == "GIN":
                        pre_train_model = GIN(train_config)
                    pre_train_model = pre_train_model.to(device)
                    pre_train_model.load_state_dict(torch.load(os.path.join(save_pretrain_model_dir, 'best.pt')))
                    logger.info(
                        "num of pretrain parameters: %d" % (sum(p.numel() for p in pre_train_model.parameters())))

                    # 获取无监督预训练嵌入
                    pretrain_embedding = pre_train(pre_train_model, graph, device, train_config)

                    print("--------------------------------------------------------------------------------------")
                    print("start task ", count)

                    current_trainset = trainset[count]
                    current_valset = valset[count]
                    current_testset = testset[count]
                    trainmask = index2mask(current_trainset, nodenum)
                    valmask = index2mask(current_valset, nodenum)
                    testmask = index2mask(current_testset, nodenum)
                    nodelabel = nodelabel.to(device)
                    pretrain_embedding = pretrain_embedding.to(device)
                    trainmask, valmask, testmask = trainmask.to(device), valmask.to(device), testmask.to(device)
                    trainlabel = torch.masked_select(nodelabel, torch.tensor(trainmask, dtype=bool)).unsqueeze(1)
                    vallabel = torch.masked_select(nodelabel, torch.tensor(valmask, dtype=bool)).unsqueeze(1)
                    testlabel = torch.masked_select(nodelabel, torch.tensor(testmask, dtype=bool)).unsqueeze(1)
                    trainemb = mask_select_emb(pretrain_embedding, trainmask, device)
                    valemb = mask_select_emb(pretrain_embedding, valmask, device)
                    testemb = mask_select_emb(pretrain_embedding, testmask, device)

                    # 2.
                    teacher_embedding = None
                    teacher_train_embedding = None
                    if train_config["use_supervised_teacher"]:
                        logger.info("Training supervised teacher model...")
                        supervised_teacher = SupervisedTeacherGIN(train_config)
                        supervised_teacher = supervised_teacher.to(device)

                        teacher_embedding = train_supervised_teacher(
                            supervised_teacher, graph, nodelabel, trainmask, valmask, device, train_config, logger
                        )

                        #
                        if teacher_embedding is not None:
                            teacher_train_embedding = teacher_embedding[trainmask]

                        logger.info("Supervised teacher training completed.")

                    # 3. 初始化学生模型
                    if train_config["prompt"] == "SUM":
                        model = node_prompt_layer_sum()
                    elif train_config["prompt"] == "LINEAR-MEAN":
                        model = node_prompt_layer_linear_mean(
                            train_config["gcn_hidden_dim"] * train_config["gcn_graph_num_layers"],
                            train_config["prompt_output_dim"])
                    elif train_config["prompt"] == "LINEAR-SUM":
                        model = node_prompt_layer_linear_sum(
                            train_config["gcn_hidden_dim"] * train_config["gcn_graph_num_layers"],
                            train_config["prompt_output_dim"])
                    elif train_config["prompt"] == "FEATURE-WEIGHTED-SUM":
                        model = node_prompt_layer_feature_weighted_sum(
                            train_config["gcn_hidden_dim"] * train_config["gcn_graph_num_layers"])
                    elif train_config["prompt"] == "FEATURE-WEIGHTED-MEAN":
                        model = node_prompt_layer_feature_weighted_mean(
                            train_config["gcn_hidden_dim"] * train_config["gcn_graph_num_layers"])

                    model = model.to(device)
                    logger.info(model)
                    logger.info(
                        "num of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

                    writer = SummaryWriter(save_model_dir)
                    if train_config["update_pretrain"]:
                        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"],
                                                      weight_decay=train_config["weight_decay"], amsgrad=True)
                        pre_optimizer = torch.optim.AdamW(pre_train_model.parameters(), lr=train_config["lr"],
                                                          weight_decay=train_config["weight_decay"], amsgrad=True)
                        pre_optimizer.zero_grad()
                    else:
                        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"],
                                                      weight_decay=train_config["weight_decay"], amsgrad=True)

                    optimizer.zero_grad()
                    scheduler = None

                    best_bp_losses = {"train": INF, "dev": INF, "test": INF}
                    best_bp_epochs = {"train": -1, "dev": -1, "test": -1}
                    best_acc = {"train": -1, "dev": -1, "test": -1}

                    total_train_time = 0
                    total_dev_time = 0
                    total_test_time = 0
                    best_c_embedding = None
                    c_embedding = None

                    # 4
                    for epoch in range(train_config["epochs"]):
                        mean_reg_loss, mean_bp_loss, _time, accfold, c_embedding = train(
                            model, optimizer, scheduler, "train", device, train_config, epoch,
                            nodelabelnum, trainemb, trainlabel, logger=logger, writer=writer,
                            teacher_embedding=teacher_train_embedding  # <--- 修改: 传入筛选后的教师嵌入
                        )

                        total_train_time += _time
                        torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))

                        if train_config["update_pretrain"] == True:
                            torch.save(pre_train_model.state_dict(),
                                       os.path.join(save_model_dir, 'pretrain_epoch%d.pt' % (epoch)))

                        if accfold >= best_acc["train"] or mean_bp_loss <= best_bp_losses["train"]:
                            if accfold >= best_acc["train"]:
                                best_acc["train"] = accfold
                            if mean_bp_loss < best_bp_losses["train"]:
                                best_bp_losses["train"] = mean_bp_loss
                            best_bp_epochs["train"] = epoch
                            logger.info(
                                "data_type: {:<5s}\tbest mean loss: {:.3f}\t best acc: {:.3f}\t (epoch: {:0>3d})".format(
                                    "train", mean_bp_loss, accfold, epoch))

                        mean_reg_loss, mean_bp_loss, evaluate_results, _time, accfold, macroFfold, weightedFfold, c_embedding, none_accuracy = \
                            evaluate(model, "val", device, train_config, epoch, c_embedding,
                                     nodelabelnum, valemb, vallabel, count, task, logger=logger, writer=writer)
                        total_dev_time += _time

                        with open(os.path.join(save_model_dir, '%s%d.json' % ("val", epoch)), "w") as f:
                            json.dump(evaluate_results, f)

                        if accfold >= best_acc["dev"] or mean_bp_loss <= best_bp_losses["dev"]:
                            if accfold >= best_acc["dev"]:
                                best_acc["dev"] = accfold
                            if mean_bp_loss < best_bp_losses["dev"]:
                                best_bp_losses["dev"] = mean_bp_loss
                            best_c_embedding = c_embedding
                            best_bp_epochs["dev"] = epoch
                            logger.info(
                                "data_type: {:<5s}\tbest mean loss: {:.3f}\t best acc: {:.3f}\t (epoch: {:0>3d})".format(
                                    "dev", mean_bp_loss, accfold, epoch))

                    best_epoch = best_bp_epochs["dev"]
                    data_loaders = OrderedDict({"test": None})
                    data_loaders["test"] = testset[count]
                    model.load_state_dict(torch.load(os.path.join(save_model_dir, 'epoch%d.pt' % (best_epoch))))

                    if train_config["update_pretrain"] == True:
                        pre_train_model.load_state_dict(
                            torch.load(os.path.join(save_model_dir, 'pretrain_epoch%d.pt' % (best_epoch))))

                    mean_reg_loss, mean_bp_loss, evaluate_results, _time, acctest, macroFtest, weightedFtest, c_embedding, none_accuracy = \
                        evaluate(model, "test", device, train_config, epoch, best_c_embedding,
                                 nodelabelnum, testemb, testlabel, count, task, debug=True, logger=logger,
                                 writer=writer)

                    none_all_acc.append(none_accuracy)
                    print("testacc:", acctest)
                    acc.append(acctest)
                    macroF.append(macroFtest)
                    weightedF.append(weightedFtest)
                    print("#################################################")
                    print("total train time", total_train_time)
                    print("#################################################")

                    for data_type in data_loaders.keys():
                        logger.info("data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type,
                                                                                                        best_bp_losses[
                                                                                                            data_type],
                                                                                                        best_bp_epochs[
                                                                                                            data_type]))

            print('none_all_acc', none_all_acc)
            print('acc for 10fold: ', acc)


            none_all_acc_cpu = [item.cpu() for item in none_all_acc]
            none_all_acc = np.array(none_all_acc_cpu)


            acc = np.array(acc)

            print('none_all_acc mean: ', np.mean(none_all_acc), 'acc std: ', np.std(none_all_acc))
            print('acc mean: ', np.mean(acc), 'acc std: ', np.std(acc))