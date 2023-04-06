---
title: Exploring different methods for testing the models.
linktitle: "April 5, 2023: Exploring more datasets and metrics"
date: 2023-04-03
# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 1
---
Research in the field of deep learning applied on traffic forecasting can broughtly be categorized on the effort of trying to have models that best predicts datasets like 
the [PEMS-BAY](https://zenodo.org/record/4263971) [METRLA](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) and [taxiBJ](https://paperswithcode.com/dataset/taxibj). Where two of thems are data from sensors from the California state and the other from the city of Beijing.
Althought the development of the models in this dataset have been a great way of standarizing the resutls of all the models it can also
represent a problem to see how good these models really are considering problems of the overfitting. An idea that comes to my mind for creating 
betters and robust models for the future is to test the generalization of these models by testing them on lots of differents graphs and have a metric for them. Instead of only training the model on a spicific dataset and test it on it. Trying to have a more general testing metric where 
all the datasets are being testesd with the same model. This type of approach will be interesting in my opinion for understanding the power of creating  meta learning systems on the field. Trying to achieve few learning or zero-shot learning from the activity of analyzing  various type of different graphs and to interpolate that knowledge to others type of graphs that are not equal. I empatizes from the statements in the [paper](https://www.mdpi.com/2220-9964/12/3/100) try to propose and exploring different types of datasets is a very good idea to stop overfitting on this types of datasets. In overall there are efforsts on trying to change the bechmark for trying to have a better understanding of the problem and that if the field of traffic forecasting there must be better ways to test our models, something that can be see from the type of improvements that new models have where there  a re little improvements of 3 to 2 percents. That are not bads  results but there is always the doubd that when moving from this type of resutls there always statistics variation on the data and it can be difficult to discriminate the effectiveness of certain models  if we dont have good ways to test thems.
