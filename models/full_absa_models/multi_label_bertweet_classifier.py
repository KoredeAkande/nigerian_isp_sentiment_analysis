#Load packages
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel

#Import package for aspect sentiment prediction
import aspect_based_sentiment_analysis as absa

#Load the ABSA sentiment model
nlp = absa.load()

#List containing the different aspect categories
ASPECTS = ['price','speed','reliability','coverage', 'customer service']

#Load BerTweet tokenizer
TOKENIZER = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

#Load the BERTweet model
BERTWEET_MODEL = AutoModel.from_pretrained("vinai/bertweet-base", from_tf = True, return_dict = True)

class ISP_TweetAspectClassifier(pl.LightningModule):
    
    #Set the aspect classifier
    def __init__(self, n_classes=5, n_training_steps=None, n_warmup_steps=None, lr=2e-5):
        
        super().__init__()
        self.lr = lr
        self.n_warmup_steps = n_warmup_steps
        self.n_training_steps = n_training_steps
        self.bert = BERTWEET_MODEL
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.criterion = torch.nn.BCELoss()
        
    def forward(self, input_ids, attention_mask, labels = None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        
        loss = 0
        
        if labels is not None:
            loss = self.criterion(output, labels)
            
        return loss, output
    
    
#Load the best model from training
mlc_model = ISP_TweetAspectClassifier.load_from_checkpoint(
    "../models/absa-aspect-extraction/bertweet/ae-epoch=19-val_loss=0.33.ckpt",
    n_classes=len(ASPECTS)
)

def run(df, col_name, optimal_threshold = 0.3):
    
    """
    Function to perform ABSA on tweets using the multi-label bertweet classifier. 
    ABSA is a two-part task of aspect extraction and aspect sentiment prediction
    
    Inputs:
        - df (pd DataFrame): A pandas dataframe to perform annotation on
        - col_name (str): The specific column in the dataframe containing the tweets run absa on
        
    Output:
        - absa_df (pd DataFrame): DataFrame containing the tweets and the ABSA results
    
    """
    
    #List to store detected aspects and their sentiments
    df_list = []
    
    #Iterate through all the tweets
    for tweet in df[col_name]:
        
        #List to store the aspects detected
        aspects_detected = []
        
        #List to store the sentiment values (Positive, Negative or Neutral) for the aspects
        detected_sentiments = []
        
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
        
        #Get the model's prediction
        _, model_prediction = mlc_model(encoding["input_ids"], encoding["attention_mask"])
        model_prediction = model_prediction.detach().numpy()
        
        #Determine the aspects detected using the optimal threshold found during fine-tuning
        model_prediction = np.where(model_prediction > optimal_threshold, 1, 0)
        
        #Iterate through the model's predictions for each aspect
        for pred_idx in range(len(model_prediction[0])):
            
            #If the aspect was detected
            if model_prediction[0][pred_idx] == 1:
                
                #Note it down
                aspects_detected.append(ASPECTS[pred_idx])
                
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