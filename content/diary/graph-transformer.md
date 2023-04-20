---
title: Graph Transformer Networks
linktitle: "4- April 12, 2023: Graph Transformer Networks"
date: 2023-04-12
# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 1
---

I will try to explain the ideas behind the Gaph Transformer Network architecture. The paper is [here](https://arxiv.org/abs/2106.06218).

GTN is an architecture that should not be confused with the usual definition of a Transformer. It is not a derivation of the Transformer architecture applied to graph data; for that, you can see networks like the ASTTN, which I previously discussed in another post. Nonetheless, understanding GTN is probably the next step in keeping up with this research.

To start understanding Graph Transformer Networks, we must first understand the concept of heterogeneous graphs and how they adapt to deep learning. Standard GNNs compute graphs with the same node and edge features, called homogenous graphs. These are the default choice when working with GNNs, but there is also the possibility of working with other types of graphs, such as heterogeneous graphs. Heterogeneous graphs have nodes of different types. For example, we can have a network representing the relation between citations in research, with nodes representing authors and others representing papers. These two types of nodes can be linked by typed edges representing various relationships, such as paper-to-author, paper-to-paper, or author-to-author. All these relationships can be expressed as a set of adjacency matrices, where each matrix represents the connectivity of each node type. Mathematically speaking, we denote this as {{<math>}}${A_k}_{k=1}^{K}$, where $K = |T^e|${{</math>}}. In general, we can define {{<math>}}$T^v$ and $T^e${{</math>}} as the sets of node and edge types in our graph. The graph can be seen as {{<math>}}$G= {V,E}$, where $V$ is the set of nodes and $E$ is the set of observed edges{{</math>}}. To fully declare the nodes, we can also define two functions, {{<math>}}$f_v : V \to T^v$ and $f_e: E \to T^e$ {{</math>}}, which classify the types of nodes and edges.

![Graph Example](heterogenous_graph_example.jpg )

A great example is the one given by the figure above. We can see how different types of nodes are connected to other nodes with different types of edges, depending on the pair of connections we are working on.


The problem with heterogeneous graphs is that we must first adapt the graphs to the type of data we have, and secondly, the architectures that work with homogenous graphs do not work with heterogeneous graphs. Normally, if we work with heterogeneous graphs, we use architectures of the form of **H-GNN**, which stands for Heterogeneous Graph Neural Networks. These systems consist of training different Neural Networks for each type of node and their corresponding edges. Thankfully, the idea of Graph Transformers is much more interesting because, at their core, they extract meaningful information about the heterogeneous graphs and then convert them to homogenous graphs.

## Meta Paths

Meta paths are paths found in heterogeneous graphs, connected by heterogeneous edge types, where {{<math>}}$v_1 \to^{\tau_e(e_1)} \dots \to^{\tau_e (e_l)} v_{l+1}$ {{</math>}}, and {{<math>}}$\tau_e (e_l) \in T^e$ {{</math>}}. The trick of the forward propagation in these networks comes in the GT layers of the architecture.

![GT Layer](GT_layer.png)


The algorithm is as follows: We first select a trivial Adjacency matrix. This matrix will be convoluted by a {{<math>}} $1 \times 1${{</math>}} kernel passed thought a softmax function.

{{<math>}} $$ F(\mathbb{A}; \phi^{(k)}) = \text{conv}_{1 \times 1}(\mathbb{A}; \text{softmax}(\phi^{(k)})) $$  {{</math>}}

That can also be seen as:

{{<math>}} $$\sum_{t=1}^{|T^e|} \alpha_t^{(k)} A_t $$   {{</math>}}  

Where alpha is :



{{<math>}}
 $$\alpha^{(k)} = \text{softmax}(\phi^{(k)}) $$ 

{{</math>}}

This essentially means that for each GT block, we calculate the weighted sum of all adjacency matrices, where the weights are first passed through a softmax function. One peculiarity of the architecture is that we no longer view the connectivity of the adjacency matrices with 0 and 1 as usually represented, but as levels of connectivity with each node that will depend on the weights for each type of heterogeneous graph.

The second step in the GT layer is to multiply the calculated weighted adjacency matrix with the previous adjacency matrix, forming the previously mentioned adjacency matrix of the meta path form.

{{<math>}} $$A^{(k-1)} F(\mathbb{A}; \phi)$$ {{</math>}}



{{<math>}}  $$A_P = \left( \sum_{t_0 \in T^e} \alpha_{t0}^{(0)} A_{t0} \right) 

\left( \sum_{t_1 \in T^e} \alpha_{t1}^{(1)} A_{t1} \right) 

\dots

\left( \sum_{t_k \in T^e} \alpha_{tk}^{(k)} A_{tk} \right) 

$$ {{</math>}}


Another thing to note is that we are not creating meta paths with mere combinations of single adjacency matrices, but instead, we take into account all adjacency matrices and assign them weights to measure their importance. The only thing that changes in each GT layer are the weights of the convolutions {{<math>}} $ \phi$ {{</math>}}.
The third step is to calculate the final adjacency matrix at the kth GT layer. This will pass through the transformation:

The third step is to calculate the final adjacency matrix at the kth GT layer. This will pass through the transformation:
{{<math>}} 
$$
A^{(k)} = \left(\hat{D^{(k)}}\right)^{-1} A^{(k-1)}  F(\mathbb{A}; \phi^{(k)})
$$
{{</math>}}
Where {{<math>}} $\hat{D^{(k)}}$  is the degree matrix of  $A^{(k)} ${{</math>}}

Then, for the final step, we perform a concatenation operation for different graph structures, and to consider multiple types of meta-paths, the final output will be a tensor {{<math>}}$\mathbb{A}^{(k)} \in R^{N \times N \times C}$ {{</math>}}. The weight vector {{<math>}} $\phi^{(k)}$ of the kth GT layer will become the weight matrix $\Phi^{(k)}$ {{</math>}}, and this operation can be represented as the following tensor equation:


{{<math>}}
$$
\mathbb{A}^{(k)} = (\mathbb{D}^{(k)})^{-1} \mathbb{A}^{(k-1)} \star F(\mathbb{A}; \Phi^{(k)})
$$
{{</math>}}

where:

{{<math>}}
$$
\mathbb{A}^{(k-1)} \star F(\mathbb{A}; \Phi^{(k)}) = \bigoplus_{c = 1}^{C} A_c^{(k-1)} F(\mathbb{A}; \phi^{(k,c)})
$$
{{</math>}}

equals the concatenations of all the meta path calculated.



After calculating the kth adjacency matrix we are going to pass the it to a Graph Convolutional Network Layer that will combined the features of the nodes with the Adjancency Matrix from the GT layers. This node representation will be assigned as **Z** and will be updated with the following function:

{{<math>}}
$$
Z^{(l+1)} = f_{agg} \left( \bigoplus_{c = 1}^{C} \sigma(\tilde{D_c}^{-1} \tilde{A}_c^{(K)} Z^{(l)} W^{(l)}) \right)
$$
{{</math>}}

This is nothing more than the definition of the GCN, but with the addition that the aggregation function will not only aggregate with respect to the neighbors but also with the batch C of the different meta paths.

## FastGTN
The initial GTN architecture can be computationally expensive due to the matrix multiplications involved, which are the bottleneck in the runtime complexity of the algorithm. To address this issue, the authors developed an upgraded version of the architecture called **FastGTN**. FastGTN is designed to save computation time when performing matrix multiplications.
In FastGTN, the authors propose a two-step approach to achieve the same goal as GTN with less computational cost:
1. Generate a low-dimensional embedding of the original adjacency matrices. This is achieved by multiplying the adjacency matrices with a small set of randomly generated matrices. The low-dimensional embeddings will maintain most of the original information but with a significantly reduced dimension.
2. Perform the matrix multiplications and other operations in the GTN using the low-dimensional embeddings instead of the original adjacency matrices.
This approach reduces the computational complexity of the matrix multiplications, making FastGTN more efficient while maintaining the effectiveness of the original GTN architecture.
In conclusion, GTN and its faster version, FastGTN, offer an innovative approach to working with heterogeneous graphs by transforming them into homogeneous graphs and leveraging meta paths to capture complex relationships in the data. 

## Conclusion

The GTN architecture has potential applications in various domains, including traffic forecasting and other real-world scenarios where heterogeneous graphs can be used to represent complex systems. Nexts steps will be on seeing how to represent  Traffic Forecasting Datasets in Heterogenous datasets and Aplying these tecniques on architectures that uses attention and other mechanims related to the Transformer architecture with heterogenous Data inference.

