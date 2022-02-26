#Import general packages
import numpy as np
import pandas as pd

#Import packages for aspect extraction
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Import package for aspect sentiment prediction
import aspect_based_sentiment_analysis as absa

# #Switch file path
# import sys
# sys.path.append("../models/full_absa_models")

#Load BerTweet tokenizer
TOKENIZER = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

#Load the tokenizer for the speed model
SPEED_TOKENIZER = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

#Add the relevant speed-related tokens, so tokenizer does not split these
SPEED_TOKENIZER.add_tokens(['mbps','mb/s','kbps','kb/s','gbps','gb/s'])


#Load the best models from finetuning 
reliability_model = AutoModelForSequenceClassification.from_pretrained("../models/absa-aspect-extraction/ensemble_model/reliability_classifier/checkpoint-130")

coverage_model = AutoModelForSequenceClassification.from_pretrained("../models/absa-aspect-extraction/ensemble_model/coverage_classifier/checkpoint-150")

price_model = AutoModelForSequenceClassification.from_pretrained("../models/absa-aspect-extraction/ensemble_model/price_classifier/checkpoint-90")

speed_model = AutoModelForSequenceClassification.from_pretrained("../models/absa-aspect-extraction/ensemble_model/speed_classifier/checkpoint-150")

customer_service_model = AutoModelForSequenceClassification.from_pretrained("../models/absa-aspect-extraction/ensemble_model/customer_service_classifier/checkpoint-140")

#Load the ABSA sentiment model
nlp = absa.load()


def run(df, col_name):
    
    """
    Function to perform ABSA on tweets. This is a two-part task of aspect extraction 
    and aspect sentiment prediction
    
    Inputs:
        - df (pd DataFrame): A pandas dataframe to perform annotation on
        - col_name (str): The specific column in the dataframe containing the tweets run absa on
        
    Output:
        - absa_df (pd DataFrame): DataFrame containing the tweets and the ABSA results
    
    """
    
    #List to store detected aspects and their sentiments
    df_list = []
    
    #List containing the binary classifiers
    binary_classifiers = [('reliability',reliability_model,TOKENIZER),
                          ('price',price_model,TOKENIZER),
                          ('speed',speed_model,SPEED_TOKENIZER),
                          ('coverage',coverage_model,TOKENIZER),
                          ('customer service', customer_service_model,TOKENIZER)]
    
    #Iterate through all the tweets
    for tweet in df[col_name]:
        
        #List to store the aspects detected
        aspects_detected = []
        
        #List to store the sentiment values (Positive, Negative or Neutral) for the aspects
        detected_sentiments = []
        
        #Iterate through each of the binary classifiers
        for aspect,classifier,tokenizer in binary_classifiers:
            
            #Encode the tweet
            encoding = tokenizer.encode_plus(
                tweet,
                add_special_tokens=True,
                max_length=TOKENIZER.model_max_length,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            #Run each binary classifier on the tweet
            aspect_prediction = classifier(encoding["input_ids"], encoding["attention_mask"])
            aspect_prediction = np.argmax(aspect_prediction[0].flatten().detach().numpy())
            
            #If the aspect was predicted to be 1, record the aspect as being found in the tweet
            if aspect_prediction == 1:
                aspects_detected.append(aspect)
                
        if aspects_detected:
            #Next, carry out sentiment prediction on the aspects detected
            sentiment = nlp(tweet,aspects = aspects_detected)
            
            #Iterate through each aspect sentiment predicted results
            for senti_result in sentiment.examples:

                #Get the sentiment scores
                scores = np.array(senti_result.scores)

                #Find the max sentiment score (i.e. the predicted sentiment value)
                max_score = np.argmax(scores)

                #Record the sentiment (string) category for the aspect
                if max_score == 2:

                    detected_sentiments.append("Positive")

                elif max_score == 1:

                    detected_sentiments.append("Negative")

                else:

                    detected_sentiments.append("Neutral")


            #Add the detected aspects and sentiments from the sentence to the list
            df_list.append([tweet,aspects_detected,detected_sentiments])
            
        else:
            df_list.append([tweet,[None],[None]])
        

    absa_df = pd.DataFrame(df_list, 
                       columns=[col_name,'Detected aspects','Predicted sentiment'])
    
    return absa_df