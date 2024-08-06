# TGN-ST

Temporal Graph Learning with Application to Large-Scale Flight Traffic Prediction (Hang et al., 2024)

This reporitory contains a PyTorch implementation of the paper TGN-ST (submitted to https://techno-srj.itc.edu.kh/ and under review).

## Table of Contents

- [TGN-ST](#tgn-st)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
  - [2. Temporal Graph Learning (TGL)](#2-temporal-graph-learning-tgl)
  - [3. Flight Network Dataset](#3-flight-network-dataset)
    - [1. Dataset Overviews](#1-dataset-overviews)
    - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [4. Evaluation Methods](#4-evaluation-methods)
  - [5. Temporal Graph Learning Pipeline](#5-temporal-graph-learning-pipeline)
  - [6. Model Selection](#6-model-selection)
  - [7. Temporal Graph Network with Static Time Encoder (TGN-ST)](#7-temporal-graph-network-with-static-time-encoder-tgn-st)
  - [7. Hyperparameter Tuning](#7-hyperparameter-tuning)
  - [8. Results](#8-results)
  - [9. Conclusion](#9-conclusion)
  - [10. License](#10-license)


## 1. Introduction

Graph are everywhere!!!

<p align="center">
    <img src="images/graph_are_everywhere.png" width="300"/>
</p>

Graph applications:

1. Transportation networks. (i.e. [@Google Maps](https://deepmind.google/discover/blog/traffic-prediction-with-advanced-graph-neural-networks/))
2. Social network (i.e. [@Meta](https://research.facebook.com/publications/revisiting-graph-neural-networks-for-link-prediction/))
3. Financial (i.e. [Lecture](https://snap.stanford.edu/graphlearning-workshop/slides/stanford_graph_learning_Finance.pdf))
4. Computer vison (i.e. [Paper](https://arxiv.org/abs/2212.10207))
5. Natural Language Processing (i.e. [Paper](https://arxiv.org/abs/2106.06090))
6. Web applications (i.e. [Web Image Search](https://europe.naverlabs.com/blog/web-image-search-gets-better-with-graph-neural-networks/))
7. Computational biology (i.e. [@Deepmind](https://deepmind.google/discover/blog/alphafold-using-ai-for-scientific-discovery-2020/))
8. Recommendation systems (i.e. [@Twitter](https://towardsdatascience.com/temporal-graph-networks-ab8f327f2efe), [@Pinterest](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48))

## 2. Temporal Graph Learning (TGL)

Systems of interaction can be modeled as graph network with entities as nodes and interaction as edges or links.

Real-world network is inherited with network change evolve over time. New subfield of machine learning is emerged as promising framework to extract strutural and temporal from dynamic graph assuming relational inductive bias and temporal inductive bias, called temporal graph learning.

<p align="center">
    <img src=images/example_dynamic_graph.png width = "300">
</p>

An example of dynamic graph. The number attached to edges are timestamps. Visualized in [LynxKite](https://lynxkite.com/)


**TGL Methods Taxonomy**

Approaches | GNN-based | Sequence-Based | Graph Walk Based | None
| :---: | :---: | :---: | :---: |:---: 
Memory-based | [JODIE](https://arxiv.org/abs/1908.01207) | - | [CAWN](https://arxiv.org/abs/2101.05974) | [EdgeBank](https://arxiv.org/abs/2207.10128)**
Self-attn -based | [TGAT](https://arxiv.org/abs/2002.07962), [TCL](https://arxiv.org/abs/2105.07944), [GAT](https://petar-v.com/GAT/)* | [DyGFormer](https://arxiv.org/abs/2303.13047) | - | - |
self-attn + Memory based | [DyREP](https://openreview.net/forum?id=HyePrhR5KX), [TGN](https://arxiv.org/abs/2006.10637) | - | - | - |
MLP-based | - | [GraphMixer](https://arxiv.org/abs/2302.11636) | - | - | - |
None | [GraphSAGE](https://arxiv.org/abs/1706.02216)**, [GCN](https://arxiv.org/abs/1609.02907)* | - | - | - |

Note:  
(*) Static graph method  
(**) Pure memorization method

## 3. Flight Network Dataset

### 1. Dataset Overviews
TGB [tgbl-flight](https://tgb.complexdatalab.com/docs/linkprop/#tgbl-flight-v2) is a crowd sourced international flight network from 2019 to 2022.


Name | #Nodes​ | #Edges​ | #Steps​​
| :---: | :---: | :---: | :---:
tgbl-flight | 18,143​ | 67,169,570​ | 1,385​

<p align="center">
    <img src=images/flight_geo_layout_plot.png width=700>
</p>

Geographically visualized of sampled flight network. Visualized [LynxKite](https://lynxkite.com/)

### 2. Exploratory Data Analysis (EDA)

Due to large-scale dataset, any graph insight operations are computationally expensive. So, we perform graph subsampling as follow:

- Dynamic Graph Subsampling and Discretization

Graph subsampling perfrom random sampling on a given number of nodes. It is also feasible to give determistic node list or regional name (i.e. continent such as NA, AS, OC, SA, EU). It is futher discretized into a given period (i.e. daily, weekly, monthly, or yearly). The graph subsampling and discretization pipeline is illustrated as follow

<p align="center">
    <img src="images/graph_subsampling.png" width="400"/>
</p>

Run Code:
```{bash}
tgb/datasets/dataset_scripts/tgbl_flight_neg_generator.py
```

Note: The negative edge sampling of the validation and test set are also generated.

The resulting sub-graph can be proceed to visualize and analyze as discussed next.

- Dynamic graph visualization

To understand the dynamic and graph structure, it is required to visualized it. Large graph poses very huge challenge in analysis and visualization due to computational resource. So, the whole network is required to perform subsampling for visualization feasible.

Code to generate .csv to input into Gephi is available at `visualization/generate_gephi_viz_file.ipynb`

By running [Force Atlas 2](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679) algorithm on sampled graph, with node sized by degree and colored by continent.

Airline route during low Covid-19 hit in Feb-21. Visualized in [Gephi](https://gephi.org/):
<p align="center">
    <img src=images/flight_graph_viz_low.png width= 500>
</p>

Airline route during high Covid-19 hit in Feb-21. Visualized in [Gephi](https://gephi.org/):
<p align="center">
    <img src=images/flight_graph_viz_high.png width= 500>
</p>

Note: North America: pink; European: blue; Asia: Orange; Oceane: red; South America: green.

Temporal Graph Network video:
https://github.com/user-attachments/assets/b1a4e248-870a-4c0f-9b69-37997b5c6d1e

The Force Atlas 2 algorthm is able to group the airport based on their located region. DORD (O'Hare International Airport) has highest number of connection. During high Covid-19 cases, the network become sparser. Very few route is connected between the aiport, especially the airline that connect between the continents.

- Dynamic Graph Analysis

While topological graph visualization provide crucial structural information about the network, it is tedious to understand the evolution of the network through time. [Poursafaei et al.](https://arxiv.org/pdf/2207.10128) and [TGX](https://github.com/ComplexData-MILA/TGX) introduce a novel technique to visualize temporal graph network. 

Temporal Edge Appearance (TEA) plot illustrates the portion of repeated edges versus newly observed edges for each timestamp in a dynamic graph.

Temporal Edge Traffic (TET) plot visualizes the reocurrence pattern of edges in different dynamic networks over time.

Temporal Edge Appearance             |  Temporal Edge Traffic
:-------------------------:|:-------------------------:
![](images/tea.png)  |  ![](images/tet.png)

Heatmap of Node Degree over Time             |  Nodes vs Edge over Time
:-------------------------:|:-------------------------:
![](images/avg_degree_heatmap.png) | ![](images/avg_n_e_degree.png)

Note: the last timestamp is not one month fully aggregated, so the 
chart is relatively low.

From TEA and TET plot, it guides how to select the model as well as the evaluation method as will be discussed next.


For complete visualization, please visit `visualization/dynamic_graph_analysis.ipynb`.

## 4. Evaluation Methods

Due to sparsity of the graph, more stringent method of evaluation is required. Mean Reciprocal Rank (MRR) is most suitable for such task as suggested by [Poursafaei et al.](https://arxiv.org/pdf/2207.10128) and [TGB](https://arxiv.org/abs/2307.01026) by contrasting one positive edge vs 20 negative edges (note exist in the network). A simple negative edge sampling, MRR evaluation and calculation method is illustrated below

<p align="center">
    <img src= "images/mrr_calculation.png" width=700>
</p>

For a given prediction link, 50% of historical edges and 50% of new edge are randomly sampled. It is crucial to test the transductive learning ability as well as the Inductive learning ability of the TGL model.

## 5. Temporal Graph Learning Pipeline

As we gain insight about the data and metric to use, we ready to build a scalable pipeline for learning on temporal graph. an end-to-end TGL pipeline is build and displayed as below.

<p align="center">
    <img src=images/tgl_pipeline.png width= 400>
</p>

The script has been tested running under Python 3.10, with the following packages installed (along with their dependencies):

    py-tgb==0.9.2
    torch_geometric==2.5.2
    wandb==0.17.4
    torch==2.2.1+cu121
    torch_cluster==1.6.3+pt22cu121
    torch_scatter==2.1.2+pt22cu121
    torch_sparse==0.6.18+pt22cu121
    torch_spline_conv==1.2.2+pt22cu121
    torchaudio==2.2.1+cu121
    torchdata==0.7.1
    scipy==1.13.0
    triton==2.2.0
    scikit-learn==1.4.2
    networkx==3.2.1
    onnx==1.16.1

Since we need to perform numerous number of experiments, [wandb](https://wandb.ai/) is a great package provided us to do experiment tracking and result visualization. As we are going to use [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) framework for FX graph compilation, we need a GPU with higher cuda compatibilyty, i.e. `torch.cuda.get_device_capability() >= 7.0`, `torch >= 2.0` and `torch_geometric >= 2.5`.

## 6. Model Selection

After built TGL pipeline, we can run many experiments without burden the GPU and spent excessive amount of time. The resulting of experiment is illustrated as below.

Test MRR             |  GPU Memory
:-------------------------:|:-------------------------:
<img src=images/test_mmr_all_models.png width=400>  |  <img src=images/gpu_memory.png width=400>

Training Time             |  Validation Time
:-------------------------:|:-------------------------:
<img src=images/train_time_all_models.png width=400> | <img src=images/val_time_all_models.png width=400>


Top Performance (MRR) | Top Efficiency (Training Speed)
:-------------------------:|:-------------------------:
DyGformer: 81% | Edgebank (No Training)
GraphMixer: 80.57% | JODIE: 2.223s
CAWN: 78.47% | TGN: 2.649s
TGN: 71.01% | TCL: 5.586s
JODIE: 68.95% | GraphMixer: 6.111s

**TGN is selected to balance the training speed and performance.**

Code for running experimentations:

- TGN model:

```{bash}
python models/TGN.py --data "tgbl-flight" --num_run 3 --seed 1
```

- Non-TGN based model such as: {`DyGFormer`, `CAWN`, `GraphMixer`, `JODIE`, `TCL`}:

```{bash}
python train_link_prediction.py --dataset_name tgbl-flight --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 3 --gpu 0
```
## 7. Temporal Graph Network with Static Time Encoder (TGN-ST)

After model selection, we can gain insight into the selected model. [onnx](https://onnx.ai/) captures the FX graph and provides below visualization of [TGN](https://arxiv.org/abs/2006.10637) model for ease of understanding. Three main modules among 5 modules of TGN are delved using onnx. An example of graph compilation using `torch.compile` is illustrated in Time Encoder.

Time Encoder | Embedding | Decoder
:-------: | :-------: | :-------:
<img src= images/time_enc.png> | <img src= images/de_gnn_compiled.onnx.png> | <img src= images/link_predictor.onnx.png>

To improve the model performance, [GraphMixer](https://arxiv.org/abs/2302.11636) suggestes to used static time encoder to imporove the model training stability.

<p align="center">
    <img src="images/tgn_st_algo.png" width= 700>
</p>

To improve the model efficiency, `torch.compile` are wrapped around the `embedding` and `link_predictor` modules. `torch.compile` is an PyTorch API that solves the graph capture using TorchDynamo and paire with backend TorchInductor to compile into fast code such as OpenAI [Triton](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) code for training on GPU.

Futhermore, the TGN forward pass in algorithm 1 implemented by [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py) poses computational cost in the standard TGN forward pass (lines 
7-10). This bottleneck arises from the loop that iterates through each negative neighbor query, `memory_updater`, `embedding`, and `link_predictor` for every node. To address this and reduce computational cost, we propose the TGN-ST forward pass algorithm in Algorithm 2. TGN-ST leverages the capabilities of modern GPUs with parallel processing architectures. Unlike the standard approach, TGN-ST avoids the need for individual loops for each node's negative neighbors (lines 6-9 in Algorithm 2). Instead, it exploits the GPU's ability to perform large matrix operations simultaneously.

- Code for experimenting TGN-ST:


```{bash}
python models/TGN_ST.py --data "tgbl-flight" --num_run 3 --seed 1
```


## 7. Hyperparameter Tuning

We performed 200 random search experiments on wandb. The main observation are listed below:
- AdamW with cosine annealing is the best traininign strategy
- Standard SGD is the worst optimizer
- Leaky_relu is the worst activation function
- Batch size and patience number is directly impact the training speed

Hyperparameter Sweeping | Hyperparameter Sweeping Top Performers
:-------: | :-------:
<img src=images/hyper_sw.png> | <img src=images/hyper_sw_high.png>

To trade-off performance and efficiency, the most suitable parameter are selected and evaluated on full dataset.

- Code for running hyperparameter sweep experimentations:


```{bash}
python models/TGN_ST_Sweep.py --data "tgbl-flight" --num_run 3 --seed 1
```

## 8. Results

**TGN-ST** sets new state-of-the-art result with **72.49%** MRR with **15%** faster training and **5x** validation time.

Training | Validation
:-------: | :-------:
<img src=images/train_loss.png> |<img src=images/val_mrr.png>
<img src=images/train_per_epoch.png> |<img src=images/val_per_epoch.png>

The consistent improvement on both dataset provides evident for the generalizability of the pipeline and TGN-ST forward pass algorithm.


<table class= "center">
    <thead>
        <tr>
            <th>Datasets</th>
            <th>Model and Gains</th>
            <th>Test MRR (%)</th>
            <th>Training/Epoch (s)</th>
            <th>Validation/Epoch (s)</th>>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>Sample</td>
            <td>TGN</td>
            <td>72.94</td>
            <td>2.3743</td>
            <td>9.8546</td>
            </tr>
        <tr>
            <td>TGN-ST</td>
            <td>80.19</td>
            <td>1.7763</td>
            <td>2.2687</td>
        </tr>
        <tr>
            <td>Gain</td>
            <td>7.25</td>
            <td>0.598 (25%)</td>
            <td>7.586 (4.34x)</td>
        </tr>
        <tr>
            <td rowspan=3>Full Dataset</td>
            <td>Vanilla TGN</td>
            <td>70.50</td>
            <td>2800</td>
            <td>9485</td>
        </tr>
        <tr>
            <td>TGN-ST</td>
            <td>72.49</td>
            <td>2366</td>
            <td>1887</td>
        </tr>
        <tr>
            <td>Gain</td>
            <td>1.99</td>
            <td>434 (15%)</td>
            <td>7598 (5.02x)</td>
        </tr>
    </tbody>
</table>

Running Code:

```{bash}
python models/TGN_ST.py --data tgbl-flight --num_run 3 --seed 1 --optimizer adamw --bs 288 --lr 0.00072 --t_0 14 --t_mult 4 --wd 0.01292
```

## 9. Conclusion
Research contributions are two-fold:
- **Scalable Framework**: Establish scalable framework, performed graph subsampling, visualization, and hyperparameter tuning.

- **Model Improvement**: Proposed TGN-ST model with new state-of-the-art performance with signifincant training and evaluation speed boost.


## 10. License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/HorHang/TGN-ST/blob/main/LICENSE)