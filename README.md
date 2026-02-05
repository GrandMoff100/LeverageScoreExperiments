# LeverageScoreExperiments (BYU Student Research Conference 2026)

## Abstract

Active learning seeks to reduce labeling and training costs by "actively" selecting the most informative data points for model training. Leverage scores are a standard technique in active linear regression that quantifies the "uniqueness" of each individual data point in data set. However, leverage scores inherently depend on the embedding or feature map of the data that one has.

Modern neural networks can be viewed as adaptive feature maps that transform raw data points into learned representations from which one can construct embeddings of the data. From this perspective, the quality of the learned embedding shapes the resulting leverage score distribution. Yet, little empirical work has examined how different choices of feature map—fixed, random, or learned—affect the behavior and usefulness of leverage scores in active learning.

We examine multiple different embeddings of standard image classification datasets, to investigate how consistent leverage score rankings for underlying data points are across various embeddings, and assess the robustness of leverage scores in identifying useful subsets of data to train on.