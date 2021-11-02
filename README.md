Nigerian ISP Aspect-Based Sentiment Analysis
==============================
This repository contains the data and code utilized in my undergraduate thesis, where I conduct an aspect-based sentiment analysis of Internet Service Providers in Lagos, Nigeria, using (code mixed) Twitter data. Skip to the repo skeleton below for an overview of the project structure.

## TL; DR:

### Sentiment Analysis
Although a fine-tuned M-BERT results in an overall improvement in predictive performance (see table below) compared to a simplified NLP library, in Textblob, the model does a poorer job predicting the minority class in our imbalanced dataset compared to the Textblob. More specifically, the M-BERT scores 0% on recall, precision, and f1 for positive samples, compared to 44% in the Textblob implementation.

|               Metric               | M-BERT Implementation | Initial Sentiment Analysis Implementation (Textblob) |
|:----------------------------------:|:---------------------:|:----------------------------------------------------:|
| Precision (class-weighted average) |          68%          |                          58%                         |
|   Recall (class-weighted average)  |          74%          |                          54%                         |
|  F1 Score (class-weighted average) |          70%          |                          52%                         |
|              Accuracy              |          74%          |                          54%                         |

This highlights the importance of accounting for the imbalance in our modelling. Current strategies being explored include:
- Reweighting the samples in the loss function to penalize more for positive samples misclassification `WIP`
    - Here I am exploring reweighting the classes in the loss function using the Effective Number of Samples (ENS) strategy, which works well in cases of extreme class imbalance (Ishan Shrivastava, 2020)
- Oversampling (or sampling with replacement) so that positive cases are more represented in the dataset
- Trying other algorithms that are known to do well with imbalanced datasets (e.g. decision trees, random forests, etc.)

### Aspect-Based Sentiment Analysis
I am conducting a very simplified aspect-based sentiment analysis with the goal of having a starting point for annotating a dataset for a more complex aspect-modeling task. This adopts an unsupervised approach, leveraging the Aspect-based Sentiment Analysis package by ScalaConsultants. Specifically, aspect terms (e.g. fast and slow) anchored to a broader aspect (e.g. speed) are fed into the model and used to determine the sentiment of the broader aspect given the aspect term(s) exists in the tweet. More on the process is outlined below.


Repository Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks, numbered for ordering
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
