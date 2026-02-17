# LeverageScoreExperiments (BYU Student Research Conference 2026)

Experiments for the talk *Assessing Representation Sensitivity of Leverage Scores in Active Learning* presented at BYU's Student Research Conference 2026.

## Abstract

Active learning seeks to reduce labeling and training costs by "actively" selecting the most informative data points for model training. Leverage scores are a standard technique in active linear regression that quantifies the "uniqueness" of each individual data point in data set. However, leverage scores inherently depend on the embedding or feature map of the data that one has.

Modern neural networks can be viewed as adaptive feature maps that transform raw data points into learned representations from which one can construct embeddings of the data. From this perspective, the quality of the learned embedding shapes the resulting leverage score distribution. Yet, little empirical work has examined how different choices of feature map—fixed, random, or learned—affect the behavior and usefulness of leverage scores in active learning.

We examine multiple different embeddings of standard image classification datasets, to investigate how consistent leverage score rankings for underlying data points are across various embeddings, and assess the robustness of leverage scores in identifying useful subsets of data to train on.

## Slides

1. Title Slide
2. What is Active Learning? What is active linear regression?
3. Introduce Leverage Scoring technique.
4. Neural Networks as Feature Maps + my big theme -> How well do leverage scores perform as an importance sampling technique?
5. question 1 - do leverage scores in different basises pick similar digits? (??????)
6. question 2 - how useful are leverage scores? Approach to training to networks (Look at performance of top-k subset selection.)
7. Back to the big question (what do we take away? They aren't useful for finding good training points because they focus on finding outliers) + Questions



Train on the same architecture, and smaller network (LiNet).
=> Do the leverage scores at the end training indicate the most useful data points to train the network in the first place?
=> Are the leverage scores indicative of important training points?
