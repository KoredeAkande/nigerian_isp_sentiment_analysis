#Import packages for cleaning
import re
import nltk
import emoji
import pandas as pd
from cleantext import clean

def clean_tweet(text,no_punc=False,no_emoji=False,no_isp_name=False):
    
    """
    Function to clean a single tweet leveraging the clean text package
    
    Input:
        - text (str): Uncleaned tweet
        
    Output:
        - cleaned_text (str): Cleaned tweet
        
    Note: Since BERT was trained on sentences, I do not remove things like punctuations or 
          numbers by deafult as it should be capable of dealing with those 
    """
    
    #Remove RT at the start of the tweet
    cleaned_text = re.sub(r"^RT","",str(text))
    
    
    cleaned_text=clean(cleaned_text,
                       fix_unicode=True, # fix various unicode errors
                       to_ascii=True,    # transliterate to closest ASCII representation
                       lower=True,       # lowercase text
                       no_line_breaks=True, # fully strip line breaks
                       no_urls=True,      # replace all URLs with ''
                       no_emails=True,   # replace all email addresses with ''
                       no_phone_numbers=True, # replace all phone numbers with ''
                       no_currency_symbols= True, # replace all currency symbols with ''
                       no_numbers=False, # replace all numbers with ''
                       no_digits=False,  # replace all digits with ''
                       no_punct= no_punc, # fully remove punctuation
                       replace_with_url="",
                       replace_with_email="",
                       replace_with_phone_number="",
                       replace_with_number="",
                       replace_with_digit="",
                       replace_with_currency_symbol="",
                       lang="en"
                      )
    
    #Remove @ symbol
    cleaned_text = cleaned_text.replace('@','')
    
    #Remove emojis
    if no_emoji:
        
        #Encode text to find emojis
        text = cleaned_text.encode('utf8')
        #Remove emojis
        cleaned_text = emoji.get_emoji_regexp().sub(r'', text.decode('utf8'))
        
#     if no_isp_name:
        
#         #Remove ISP names –– UPDATE IF YOU ADD MORE ISPS!
#         cleaned_text = re.sub('\b\w*?ranet\w*?\b|\b\w*?tizeti\w*?\b|\b\w*?ipnx\w*?\b','',cleaned_text)
    
    return cleaned_text


def run_cleaner(df,column,no_punc,no_emoji,no_isp_name):
    
    """
    Function to perform cleaning operations on a dataframe's column
    """
    
    #Clean the tweets column
    cleaned_column = df[column].apply(clean_tweet,
                                      no_punc = no_punc,
                                      no_emoji = no_emoji, 
                                      no_isp_name = no_isp_name)
    
    
    #Return the dataframe with the cleaned column
    df[column] = cleaned_column
    
    
    return df
    
