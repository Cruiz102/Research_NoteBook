---
title: Positional Embedding on Graphs.
linktitle: "3- April 6, 2023: How positional Embedding is applied to Graph Neural Networks."
date: 2023-04-06
# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 1
---
Positional embedding is a core component in the current development of machine learning. The Transformer architecture has led to significant advancements in numerous areas, most notably in NLP applications with Large Language Models (LLMs), such as the GPT models.

The original Transformer architecture was published by Vaswani et al. in 2017. One of the most important ideas in this paper is that of positional embedding and how to use only the attention mechanism for sequence data problems. This marked a revolution in managing sequence data, as before the Transformer, the state of the art was Recurrent Neural Networks. However, the way RNNs process sequential data limits the parallelization of model training, and requires significant time for calculating gradients through a process called Backpropagation Through Time (BPTT). In BPTT, the backpropagation process cannot be parallelized becuase of charateristic that the forward propagation in RNN cannot be batched.

![Image of the Transformer Architecture](transformer.png 'Original Transformer architecture Diagram')

In this post, I will explain the positional embedding algorithm, the underlying math, and how spectral graph theory can be used to extrapolate the positional embedding in graphs.

The positional Embedding formula is the following.

{{< math >}}
$$
\vec{v_t^{(i)}} = \cases{\sin(w_k \cdot t) \text{   if i = 2k}\\ \cos(w_k \cdot t) \text{ if  i = 2k + 1}} 
$$
$\text{Where} \\$
$$ w_k = \frac{1}{10000^{k/d}}$$
{{< /math >}}

The formula originates from the idea of representing the position of the data by concatenating a binary number representation or, in the case of the Transformer, adding a positional vector to the data. The core idea that made the process as smooth as possible was using sinusoidal functions to represent the positional embedded vectors. This approach has many advantages, as neural networks work with floating values and functions that attempt to simulate continuous representations. Sines and cosines can represent bounded values effectively and provide significant improvements for representing positional information. Discrete representations, like representing positions with zeros and ones, can be repetitive for the model to understand and cannot represent position as effectively, as they do not imitate the nature of how neural networks work with backpropagation.

### A little about espectral theory

Graphs are essential objects of study in various applications, leading mathematicians and professionals in the area to develop methods for understanding these mathematical objects. One such method is called Spectral Theory, which stems from the idea of analyzing the connectivity information of the objects. The Laplacian Matrix, defined as {{< math >}}$L = D - A${{< /math >}} (where D is the degree matrix and A is the adjacency matrix of the graph), is calculated. However, for the purpose of GNNs, we use the following equivalent definition {{< math >}}$L = I - D^{1/2}AD^{1/2}$ {{< /math >}}. The reason for choosing this definition over the more official one is that it considers the normalized information of the matrix, which is more stable to compute. We then calculate the Eigen Decomposition of the Laplacian Matrix {{< math>}} $LV = V\Lambda ${{< /math>}}. Since the Laplacian Matrix has some beneficial properties like its symmetry (due to the structure of the adjacency matrix for undirected graphs and the property that {{<math>}} $L = \hat{A}\hat{A}^T$ {{</math>}} where hat A is the incidence matrix), its eigenvectors are always orthogonal. We can normalize them to obtain orthonormal matrices with the property that {{<math>}} $V V^T = I$ {{</math>}}. This results in another Eigen Decomposition called spectral decomposition, where {{<math>}} $L = V\Lambda V^T$ {{</math>}}. When we multiply the eigenvectors with a vector {{<math>}} $ x${{</math>}}, which we will call a signal, we obtain the following:
{{<math>}}
$$
X = V^T x
$$
{{</math>}}

This operation is called the **Graph Fourier** Transform. It is the equivalent of the Discrete Fourier Transform but applied to graph-structured data instead of well-ordered Euclidean data like wave signals or images. From this operation, we extract the positional information of the graph and, based on the original positional embedding formula, add it to the feature information of the nodes. For applications in temporal-spatial problems like traffic forecasting or analyzing videos with graphs, we can add a module that integrates time, as demonstrated by the [paper](https://arxiv.org/abs/2207.05064) that introduced the ASTTN architecture, which incorporates temporal features into the calculations of the positional embedding.

![Photo of the ASTTN](asttn.png)

A brief description of the usage of the spectral information of the graph in the positional embedding is as follows: If we order the eigenvectors so that the eigenvalues are in ascending order, we can select the first k values of the matrix. We choose the lowest values because they hold information about localized graph structures in the Laplace Matrix. As shown in the figure below, while the lowest values hold more localized information, the largest values are more smoothly associated with the eigenvalues and have the most information about the entire graph, giving us a very abstract first approximation of the matrix.
![positional embedding](pe.jpg)

### Ideas and Conclusion

Spectral GNN is a topic worth studying for a deeper understanding of advancements in the field. Concepts like Spectral Clustering and understanding why current approaches to Graph Convolutional Layers are structured the way they are, are crucial for expanding knowledge of GNNs. In the context of Graph Transformers, using this definition of positional embedding, I like to think that is a combination of Graph Convolution Graph(GCN) tecniques and Graph Attention Layers(GAT). There are many possibilities for altering architectures like ASTTN, such as other proposed positional embedding functions based on common neighborhoods, but it is likely that these have already been explored. As I learn more about these architectures, I am increasingly interested in working with heterogeneous and dynamic graphs rather than spatial-temporal homogeneous graphs. I believe that the power of the Transformer, combined with the generality of heterogeneous dynamic graphs, is a potent combination to explore and investigate further.

