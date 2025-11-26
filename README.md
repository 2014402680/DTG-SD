# DTG-SD: Dual-Teacher Guided Structured Distillation for Multi-Task Drug Discovery
## Introduction:
Therefore, we propose a dual-teacher structured dynamic knowledge distillation
framework, DTG-SD, to enable efficient knowledge transfer between upstream and
downstream tasks and thereby achieve accurate multi-task drug association analysis. First, we construct a multi-task general teacher model FT based on the Task-Adaptive
Parameter Optimizer (TAPO), which adaptively updates the parameters of key tasks
in each iteration to maximize cross-task knowledge sharing and alleviate task
conflicts. Subsequently, DTG-SD adopts a dual-teacher cascading mechanism, where the
general knowledge learned by FT further guides the training of a ST using only a
small number of labeled samples, forming task-aligned internal representations and
enabling effective task-customized knowledge transfer. Finally, we design a structured
knowledge distillation strategy in which the SM learns not only the probability
distribution of ST but also aligns with the geometric structure of ST's embedding
space through interactive prototype generation and teacher-guided calibration. Unlike
traditional imitation-based distillation, this approach promotes sufficient and
interpretable knowledge transfer via teacher–student interactive structural alignment, resulting in more generalizable and discriminative task representations.

## Requirements

```
python = 3.9
dgl-cuda11.3 = 0.9.1
numpy = 1.25.0
torch-scatter = 2.1.0+pt112cu113
torch = 1.12.1
```

## Overview

```
├── get_nodefeature_module                # Data processing, constructing graph data required for training.
├── prompt_module                         # Graph pre-training, subgraph representation.
│   ├── load_down_train_node_data         # Perform ten-fold cross-validation on the data.
│   ├── node_data                         # Stored node training data
│   │   ├── luo                           # Luo’s data
│   │   ├── zheng                         # Zheng’s data
│   ├── prompt_embedding                  # Stored hint vectors
│   ├── pretrain.py                       # Model pre-training
│   ├── run.py                            # training code 

```

## Examples Instructions
Take the Zheng’s dataset for example.
 1. Constructing heterogeneous graph:
```
 ./get_nodefeature_module/test/random_vector_generation_10000_zheng.py
 ./get_nodefeature_module/graph_dti/shuffle_save_down_graph.py
```
2. Graph Pre-training:
```
./prompt_module/pretrain.py
```
3. Sub-graph Representation(Pre-trained weights are saved; you can run this directly):
```
./prompt_module/run.py
```


