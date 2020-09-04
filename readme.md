# Siamese Networks for One-Shot Learning

A Pytorch re-implementation of the [original paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) with
training and testing on the custom face dataset to determine facial similarities.

## Approach
Siamese Neural Networks take two samples as input and outputs a probability for whether given inputs belong to the same class or not. 
Input samples pass through identical CNNs (with shared weights), and their embeddings are compared in the cost function.
Here, a **contrastive loss function** is used to find the similarity between the image pair. 