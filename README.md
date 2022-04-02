Nigerian ISP Aspect-Based Sentiment Analysis
==============================
This repository contains the data and code utilized in my undergraduate thesis, where I conduct an (aspect-based) sentiment analysis of Internet Service Providers in Lagos, Nigeria, using Twitter data. The project culminates in a [Tableau dashboard](https://public.tableau.com/app/profile/korede.akande/viz/SpectranetDashboardV4/SpectranetMain) to inform understanding of customers and facilitate strategic decision making. Skip to the `Repository Organization` below for an overview of the project structure.

Sentiment Analysis
------------
Three models were experimented with for Nigerian Internet Service Providers' sentiment analysis, based on two factors: *Multilingualism* and *Proximity to Problem Domain (i.e. Twitter)*. The specific models fine-tuned (and their justification) include:
- [BERTweet](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) (Proximity to Problem Domain)
- [Multilingual-BERT](https://huggingface.co/bert-base-multilingual-cased) (Multilingualism)
- [XLM-roBERTa-base](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) (Multilingualism & Proximity to Problem Domain)

After experimenting with reweighting the loss function, increasing the batch size, and oversampling the minority to account for class imbalance, and experimenting with weight decay to address overfitting, the best checkpoints for the different models obtained the following validation results

|       Model      | Accuracy     | Precision | Recall |  F-1  |
|:----------------:|:------------:|:---------:|:------:|:-----:|
|      M-BERT      |   65.8%      |   59.8%   |  49.9% | 51.8% |
|     BERTweet     |   **85.5%**  |   81.9%   |  **84.3%** | **83.0%** |
| XLM-roBERTa-base |   82.9%      |   **87.3%**   |  72.4% | 77.0% |
**Note:** All metrics above are macro-averaged

Aspect-Based Sentiment Analysis (ABSA)
------------
ABSA can be broken down into two subtasks: Aspect Extraction (AE) and Aspect Sentiment Classification (ASC)

#### Aspect Extraction (AE)
The aspect extraction subtask was framed as a multi-label classification problem. Hence, a single tweet can have multiple aspects (in our case one or more out of *price, speed, coverage, customer service*, and *reliability*). To tackle the multilabel classification problem, three approaches were compared:

1. POS Tagging with word similarity 
2. Multi-label classification with fine-tuned BERTweet model
3. Binary relevance classification (using an independent BERTweet classifier for each aspect/label)

F-0.5 score (which puts twice as much weight on precision than recall) was chosen as the key metric. The validation results obtained from these models are thus:

|                        Model | Price F-0.5 | Speed F-0.5 | Reliability F-0.5 | Coverage F-0.5 | Customer service F-0.5 |
|:-----------------------------:|:------------:|:------------:|:------------------:|:---------------:|:-----------------------:|
| POS tagger + word similarity |        0.0% |       29.4% |              0.0% |           0.0% |                  17.9% |
|             Binary relevance |       **80.4%** |       **84.6%** |             **75.7%** |          **85.4%** |                  **78.8%** |
|         Multi-label BERTweet |       20.8% |       34.5% |              8.5% |          38.5% |                  65.8% |


#### Aspect Sentiment Classification (ASC)
Following the determination of the best aspect extraction model (Binary relevance), aspect sentiment classification was carried out using the [Aspect-based Sentiment Analysis](https://github.com/ScalaConsultants/Aspect-Based-Sentiment-Analysis) package by ScalaConsultants. The accuracy results for the different aspects is thus

|          | Price | Speed | Reliability | Coverage | Customer Service |
|:--------:|:-----:|:-----:|:-----------:|:--------:|:----------------:|
| Accuracy | 22.2% | 63.6% |     100%    |   100%   |       90.9%       |


Installation & Quick Start
------------
1. Download the models
- **Sentiment analysis model:** Download the [checkpoint](https://drive.google.com/drive/folders/11CJV9Zv5IVJuLQXRZUIRLf1gZshmCNY1?usp=sharing) and place inside `models/bertweet/baseline-bertweet`.
- **ABSA models:** Download the `best-checkpoints` [folder](https://drive.google.com/drive/folders/1ckFSG45S96NPXn_EEnvR9DyG51cSg_Ek?usp=sharing), rename it `ensemble_model` and place inside `models/absa-aspect-extraction` folder. Note: The folder is large!
2. Install requirements

```
pip install -r requirements.txt
```
3. Sentiment analysis quick start

```
#Load the sentiment analysis model
sys.path.append("../models/sentiment_analysis_models")
import bertweet_sentiment_model

#Compute sentiment analysis results
sentiment_results = bertweet_sentiment_model.run(data,'Text')

```

4. Aspect-based sentiment analysis quick start

```
#Load the absa model
sys.path.append("../models/full_absa_models")
import binary_relevance_model

#Compute ABSA results
absa_results = binary_relevance_model.run(data,'Text')

```

Repository Organization
------------

    ├── LICENSE
    ├── README.md               <- The top-level README for this project.
    ├── data
    │   ├── analogous-data      <- Data different but related to the Nigerian ISP domain
    │   ├── interim             <- Intermediate data that has been transformed.
    │   ├── processed           <- The final, canonical data sets for modeling.
    │   ├── dataset-graveyard   <- Trash. Place to store work that is currently unimportant but could be useful
    │   ├── model-evaluation    <- Datasets for evaluating the generated models
    │   ├── model-generated     <- Datasets created by the models
    │   └── raw                 <- The original, immutable data dump.
    │
    ├── model_logs              <- Logs from model runs
    │
    ├── models                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks               <- Jupyter notebooks, numbered for ordering
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures             <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                              generated with `pip freeze > requirements.txt`
    │
    ├── src                     <- Source code for use in this project.
    │    └── credentials        <- Twitter API credentials
    │
    └── py_scripts
        ├── absa_metrics.py     <- Script containing metrics for aspect-based sentiment analysis evaluation
        └── clean_tweets.py     <- Script to clean tweets
        
--------



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
