# Hyybrid Recommendation System

This project is implementation of a hybrid recommender system:
* Model-based Collaborative-filtering technique in PyTorch
* Bayesian Average method.

the Model-based Collaborative-filtering technique uses non-negative matrix factorization and is applied for existing users. In case of adding new user to database without any historical data (cold start), the Bayesian Average has been applied.

The project is trained on [MovieLens 100k](https://grouplens.org/datasets/movielens/) dataset.