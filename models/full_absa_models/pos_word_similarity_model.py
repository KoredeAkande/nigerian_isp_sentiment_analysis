
#Import relevant packages
import nltk
import numpy as np
import pandas as pd
import aspect_based_sentiment_analysis as absa

from nltk import pos_tag, RegexpParser

#Packages for word relatedness computation
import spacy
spacy_nlp = spacy.load('en_core_web_lg')

#Load the ABSA model to perform the annotation
nlp = absa.load()

#1. List aspects determined during the annotation phase
    #Note: This might not be exhaustive! But it should cover most cases. It is also subjective!
    #Also using synonyms of these words will likely yield different results
aspects = ['price','speed','reliability','coverage', 'customer service']

#2. Pair aspects with their tokenized form to avoid recomputation in the ABSA phase below
aspects_with_token = [] #List to store the pairing

#Iterate through the aspects and compute their word vector using spacy
for aspect in aspects:
    aspects_with_token.append((aspect,spacy_nlp(aspect)))
    
def run(df, col_name, similarity_threshold = 0.6):
    
    """
    Function to perform unsupervised annotation of tweets based on process outlined above
    
    Inputs:
        - df (pd DataFrame): A pandas dataframe to perform annotation on
        - col_name (str): The specific column in the dataframe containing the tweets to use for annotation
        - similarity_threshold (float): The threshold for aspect detection
        
    Output:
        - absa_df (pd DataFrame): DataFrame containing the tweets and their ABSA annotation (if relevant)
    
    """
    
    #Set to store all seen words
    seen_words = set()

    #Set to store all aspect implying words found – to avoid recomputing similarity scores
    aspect_implying_words_glob = set()

    #Dictionary categorizing all aspect-implying words into their relevant aspects
    aspects_with_implying_words = {'price':set(),'speed':set(),'reliability':set(),'coverage':set(), 
                                   'customer service':set(),'trustworthiness':set()}

    #List to store detected aspects and their sentiments
    df_list = []

    #Similarity threshold
    sim_thresh = similarity_threshold

    #Chunk tags to match – i.e. parts of speech to extract
    CHUNK_TAG = """
    MATCH: {<NN>+|<NN.*>+}
    {<JJ.*>?}
    {<RB.*>?}
    """

    #Initialize chunk tag parser
    cp = nltk.RegexpParser(CHUNK_TAG)

    #Iterate through all the tweets
    for tweet in df[col_name]:

        #Set to store the detected aspects at the sentence level
        # detected_aspects = set()

        #Dictionary to store the sentiment value for each seen aspect
        sentence_lvl_aspect_sentiment = {'price':[],'speed':[],'reliability':[],'coverage':[], 
                                         'customer service':[], 'trustworthiness':[]}

        #Split the tweet into words
        text = tweet.split()

        #Tag the words with their part of speech
        tokens_tag = pos_tag(text)

        #Get the words with relevant POS (noun, adverbs, adjectives)
        chunk_result = cp.parse(tokens_tag)

        #Extract chunk results from tree into list 
        chunk_items = [list(n) for n in chunk_result if isinstance(n, nltk.tree.Tree)]

        #Finally fuse/extract chunked words to get (noun) phrases, nouns, adverbs, adjectives
        #1. List to store the words
        matched_words = []

        #2. Iterate through the chunked words lists and get the relevant words
        for item in chunk_items:
            if len(item) > 1:
                full_string = []

                for word in item:
                    full_string.append(word[0])

                matched_words.append(' '.join(full_string))

            else:
                matched_words.append(item[0][0])

        #Iterate through all the words
        for word_in_focus in matched_words:

            #If the word has been seen before
            if word_in_focus in seen_words:

                #Check if the word is an aspect-implying word
                if word_in_focus in aspect_implying_words_glob:

                    #List to store all the aspects found to related to the certain word/token
                    aspects_implied = []

                #If it is an aspect-implying word, iterate through all the aspects
                for aspect in aspects_with_implying_words.keys():
                    
                    #Check if the word_in_focus was noted as a word implying the aspect
                    if word_in_focus in aspects_with_implying_words[aspect]:
                        
                        #Get all the aspects the word_in_focus implies
                        aspects_implied.append(aspect)
                        
            
            else:
                continue
                    
         
        #If the word hasn't been seen before
        else:
            
            #Mark the word as seen now
            seen_words.add(word_in_focus)
                
            #List to store all the aspects found to related to the certain word/token
            #Ideally a given word won't imply multiple of the aspects as they are fairly independent
            #-but just in case 
            aspects_implied = []

            #Iterate through all the aspects
            for aspect,asp_token in aspects_with_token:

                #Translate word_in_focus to word vector
                spacy_token = spacy_nlp(word_in_focus)

                #Compute the similarity between the two word vectors (i.e. the two words)
                #Round up to 1 d.p.
                similarity_score = round(asp_token.similarity(spacy_token),1)

                #If the max similarity score seen is greater than the threshold
                if similarity_score >= sim_thresh:

                    #Add the word to the set of all aspect-implying words seen
                    aspect_implying_words_glob.add(word_in_focus)

                    #Add the word to the dictionary of the relevant aspect word
                    aspects_with_implying_words[aspect].add(word_in_focus)

                    #Note that the aspect has been found in this particular sentence
                    # detected_aspects.add(aspect)

                    #Add the aspect to the list of aspects that the word_in_focus implies
                    aspects_implied.append(aspect)


                #If the word is not an aspect implying word, continue to next word
                else:

                    continue
                
        #Calculate the sentiment scores for the aspect_implying word in the current sentence
        sentiment = nlp(tweet ,aspects = [word_in_focus])
        sentiment_scores = sentiment.subtasks[word_in_focus].examples[0].scores

        #Note down the scores for all the implied aspects
        for aspect in aspects_implied:
            sentence_lvl_aspect_sentiment[aspect].append(sentiment_scores)
    
        #List to store the detected aspects from the sentence
        detected_aspects = []

        #List to store the determined sentiments of the detected aspects
        detected_sentiments = []

        #Iterate through all the aspects
        for aspect in sentence_lvl_aspect_sentiment.keys():

            #If the aspect was detected in the sentence
            if sentence_lvl_aspect_sentiment[aspect]:

                #Record this
                detected_aspects.append(aspect)

                #Calculate the average sentiment scores across the different terms
                avg_senti_score = np.array(sentence_lvl_aspect_sentiment[aspect]).mean(axis=0)

                #Get the sentiment category (neutral,negative,positive) with the largest probability
                max_idx = np.argmax(avg_senti_score)

                if max_idx == 2:

                    detected_sentiments.append("Positive")

                elif max_idx == 1:

                    detected_sentiments.append("Negative")

                else:

                    detected_sentiments.append("Neutral")

        #Add the detected aspects and sentiments from the sentence to the list
        if detected_aspects:
            df_list.append([tweet,detected_aspects,detected_sentiments])
        else:
            df_list.append([tweet,[None],[None]])


    absa_df = pd.DataFrame(df_list, 
                       columns=[col_name,'Detected aspects','Corresponding sentiment'])
    
    return absa_df