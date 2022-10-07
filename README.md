# IMDB Bayesian Logistic Regression Model
Applying Bayesian Logistic Regression to IMDB 50K Dataset (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Logistic Regression Implementation

### Binary classification problem

There are only two classes (positive, negative) used for sentiment analysis of the 50K IMDB dataset. Hence, we aim to apply Bayesian logistic regression to perform MAP inference and predict the sentiment scores from the test data. 

### Feature representation of reviews
To perform Bayesian logistic regression, we must first decide on the features to be used to represent each review. Each review consists of a body of text that we can extract features from (ex. count of negative words). We use the Bag of Words assumption to simplify our task (model does not need to capture dependencies between input). 

Traditionally, we can use simple count vectorization and apply one-hot encoding. However, one-hot encodings have the following disadvantages:
- BOW  cannot count for unobserved sentences (increasing vocabulary)
- data is very sparse considering every data sample likely covers only a small proportion of the vocabulary

### Training through SGD
Then we will perform a linear transformation on these features using an initial set of weights. We will then apply the logistic function (sigmoid) which should map the values of the transformation as probabilities from 0 to 1. Based on the probability generated, we can assign a sentiment value and calculate the loss based on the true label.



## Python Implementation Notes

#### [Indexing a Pandas Dataset](https://re-thought.com/how-to-change-or-update-a-cell-value-in-python-pandas-dataframe/)
- `loc`: selects subsets of rows and columns by label only
- `iloc`: selects subsets of rows and columns by integer location only
- `at`: selects a single scalar value in the DataFrame by label only
- `iat`: selects a single scalar value in the DataFrame by integer location only

#### Scikit-Learn parameters
- `random_state`: sets seed, makes random generator deterministic (any integer suffices)