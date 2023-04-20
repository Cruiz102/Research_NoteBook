---
title: Exploring different methods for testing the models.
linktitle: "2- April 5, 2023: Exploring more datasets and metrics"
date: 2023-04-05
# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 1
---
Research in the field can broadly be categorized as the effort to develop models that best predict datasets such as [PEMS-BAY](https://zenodo.org/record/4263971#.ZDA_invMK3A), [METR-LA](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX), and [taxiBJ](https://paperswithcode.com/dataset/taxibj). Two of these datasets are derived from sensor data in California, while the third is from the city of Beijing.

Although using these datasets to develop models has been an effective way to standardize results, it may also present challenges in determining the true performance of the models, given the potential for overfitting. One approach to creating better, more robust models for the future is to test their generalization capabilities on a variety of different graphs and develop a suitable metric. Instead of training and testing the model on a specific dataset, a more general testing metric could be employed where all datasets are tested using the same model. This approach may be valuable in understanding the potential for creating meta-learning systems in the field, which could facilitate few-shot or zero-shot learning by analyzing various types of graphs and extrapolating that knowledge to other graph types.

I empathize with the statements in this [paper](https://www.mdpi.com/2220-9964/12/3/100), which suggests that exploring diverse datasets is a good strategy to prevent overfitting. Overall, there are efforts being made to change the benchmark in order to gain a better understanding of the problem. In the field of traffic forecasting, there is a need for improved methods to test our models. Current model improvements show incremental gains of 2-3%, which are not inherently negative results. However, there is always the doubt that these improvements could be attributed to statistical variations in the data. It becomes difficult to discern the true effectiveness of specific models without reliable testing methods.