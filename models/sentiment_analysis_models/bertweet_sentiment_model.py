import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Load BerTweet tokenizer
TOKENIZER = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis", normalization=True)

#Load the BERTweet model
BERTWEET_MODEL = AutoModelForSequenceClassification.from_pretrained("../models/bertweet/baseline-bertweet/checkpoint-203")

def run(df, col_name):
    
    """
    Function to perform sentiment analysis on tweets.
    
    Inputs:
        - df (pd DataFrame): A pandas dataframe to perform annotation on
        - col_name (str): The specific column in the dataframe containing the tweets run sentiment analysis on
        
    Output:
        - absa_df (pd DataFrame): DataFrame containing the tweets and the sentiments
    
    """
    
    #List to store the tweet sentiments
    tweet_sentiments = []
    
    #Iterate through all the tweets
    for tweet in df[col_name]:
        
        #Encode the tweet
        encoding = TOKENIZER.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=TOKENIZER.model_max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        #Get the prediction 
        sentiment_prediction = BERTWEET_MODEL(encoding["input_ids"], encoding["attention_mask"])
        sentiment_prediction = np.argmax(sentiment_prediction[0].flatten().detach().numpy())
        
        #Convert to string sentiment labels
        if sentiment_prediction == 0:
            
            tweet_sentiments.append('Negative')
        
        elif sentiment_prediction == 1:
            
            tweet_sentiments.append('Neutral')
            
        else:
            
            tweet_sentiments.append('Positive')
            
        
    #Turn into a dataframe
    sentiment_df = pd.DataFrame(tweet_sentiments, 
                   columns=['Sentiment predictions'])


    return sentiment_df