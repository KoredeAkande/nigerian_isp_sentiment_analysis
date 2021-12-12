import numpy as np

def weighted_binary_precision_recall_fscore(true_aspects,aspect_preds, beta = 1):
    
    """
    Function to compute the support-weighted precision, recall and f-score based on the model's predicitions
    
    Formulas guided by
        - Jason Brownlee. (2020, February 23). A Gentle Introduction to the Fbeta-Measure for Machine Learning. 
          Machine Learning Mastery. https://machinelearningmastery.com/fbeta-measure-for-machine-learning/

    """
    
    #Note the different aspect classes
    ASPECTS = ['price','speed','reliability','coverage', 'customer service']
    
    #Dictionary to note the number of true positives, false positives and false negatives 
    #for the different classes
    class_metrics = {}
    
    #Iterate through all the aspects
    for aspect in ASPECTS:
        
        #Initialize counters for true positives, false positives and false negatives
        TP, FP, FN, TN = 0, 0, 0, 0
        
        #Iterate through all the tweets
        for idx in range(len(aspect_preds)):
            
            #If the predicted aspect is truly in the tweet
            if (aspect in aspect_preds[idx]) and (aspect in true_aspects[idx]):
                
                #Note a true positive
                TP += 1
            
            #If the aspect is in the tweet but the model does not recognize it
            if (aspect not in aspect_preds[idx]) and (aspect in true_aspects[idx]):
                
                #Note false negative
                FN += 1
                
            #If the predicted aspect is not truly in the tweet
            if (aspect in aspect_preds[idx]) and (aspect not in true_aspects[idx]):

                #Record false positive
                FP += 1
                
            #If the aspect was correctly left out of the tweet
            if (aspect not in aspect_preds[idx]) and (aspect not in true_aspects[idx]):

                #Record false positive
                TN += 1
        
        #Calculate class level precision, recall, F1 and support
        support = TP + FN
        
        try:
            precision = float(TP/(TP+FP))
        except ZeroDivisionError:
            precision = 0
            
        try:
            recall = float(TP/(TP+FN))
        except ZeroDivisionError:
            recall = 0
        
        #Calculate class level f score
        try:
            fscore = ((1 + beta**2) * precision * recall)/(beta**2 * precision + recall)
        except ZeroDivisionError:
            fscore = 0

        #Note down the final class-level metrics
        class_metrics[aspect] = {'TP':TP, 'FP':FP, 
                                 'FN': FN, 'TN':TN,
                                 'Support': support,
                                 'Precision': precision, 
                                 'Recall': recall,
                                 f'F-{beta}': fscore}
                
        
    #COMPUTE WEIGHTED PRECISION, RECALL AND F-SCORE
    
    #Counters to track class aggregated metrics
    weighted_precision, weighted_recall, weighted_fscore = 0,0,0
    
    #Compute support across all classes
    total_support = sum([class_metrics[key]['Support'] for key in class_metrics.keys()])
    
    #Iterate through all the classes
    for aspect_key in class_metrics.keys():
        
        #Get the precision and weight by support
        weighted_precision += class_metrics[aspect_key]['Precision'] * (class_metrics[aspect_key]['Support']/total_support)
        
        #Get the recall and weight by support
        weighted_recall += class_metrics[aspect_key]['Recall'] * (class_metrics[aspect_key]['Support']/total_support)
        
        #Get the precision and weight by support
        weighted_fscore += class_metrics[aspect_key][f'F-{beta}'] * (class_metrics[aspect_key]['Support']/total_support)
    
    #Return class level metrics, weighted-precision, weighted-recall and weighted-f measure
    return class_metrics, weighted_precision, weighted_recall, weighted_fscore


def aspect_sentiment_accuracy(true_aspects,aspect_preds,true_sentiment,sentiment_preds):
    
    """
    Compute the micro and macro accuracy for the aspect sentiment predictions
    """
    
    import numpy as np
    
    #Ensure all the inputted values contain lists and not strings
    true_aspects = true_aspects.apply(lambda x: eval(x) if isinstance(x,str) else x)
    aspect_preds = aspect_preds.apply(lambda x: eval(x) if isinstance(x,str) else x)
    true_sentiment = true_sentiment.apply(lambda x: eval(x) if isinstance(x,str) else x)
    sentiment_preds = sentiment_preds.apply(lambda x: eval(x) if isinstance(x,str) else x)
    
    #Note the different aspect groups
    ASPECTS = ['price','speed','reliability','coverage', 'customer service']
    
    #Dictionary to note the sentiment prediction accuracy for the different aspects
    aspect_accuracy = {}
    
    #Track the number of correct preds and total preds across all the classes
    global_correct_preds, global_total_preds = 0, 0
    
    #Iterate through all the aspects
    for aspect in ASPECTS:
        
        #Initialize counters for number of correct predictions and number of total predictions
        correct_preds, total_preds = 0, 0
        
        #Iterate through all the tweets
        for idx in range(len(aspect_preds)):
            
            #If the predicted aspect is truly in the tweet
            if (aspect in aspect_preds[idx]) and (aspect in true_aspects[idx]):
                
                #Convert to numpy array
                model_preds = np.array(aspect_preds[idx]) #Model preds
                
                #Convert to numpy array
                true_preds = np.array(true_aspects[idx]) #True preds

                #Find the corresponding sentiment of the correctly predicted aspect
                #1. In model preds
                sentiment_pred = sentiment_preds[idx][np.argwhere(model_preds == aspect)[0][0]]
                
                #1. In true preds
                true_sentiment_pred = true_sentiment[idx][np.argwhere(true_preds == aspect)[0][0]]
                
                #If the predicted sentiment for the aspect is equal to the true sentiment
                if sentiment_pred == true_sentiment_pred:
                    
                    #Record as correct prediction
                    correct_preds += 1
                    global_correct_preds += 1 #Global case
                    
                
                #Record a prediction regardless of if correct or not
                total_preds += 1
                global_total_preds += 1 #Global case
                
                
        #Note down the final class-level accuracy
        try:
            aspect_accuracy[aspect] = correct_preds/total_preds
        except ZeroDivisionError:
            aspect_accuracy[aspect] = 'No prediction for this aspect'
            
                
    #Compute the global/micro accuracy (across all aspects)
    micro_accuracy = global_correct_preds/global_total_preds
    
    #Compute the macro accuracy (unweighted average from all the classes)
    macro_accuracy = np.mean([aspect_accuracy[aspect] for aspect in aspect_accuracy.keys() if isinstance(aspect_accuracy[aspect],float)])
    
    #Return class level metrics, micro-accuracy, and macro accuracy
    return aspect_accuracy, micro_accuracy, macro_accuracy
    