import numpy as np

def binary_precision_recall_fscore(true_aspects,aspect_preds, beta = 1):
    
    """
    Function to compute the micro-averaged precision, recall and f-score based on the model's predicitions
    
    Formulas guided by:
    
        - MICRO-PRECISION:
          Micro-precision on the Peltarion Platform. (2021). Micro-precision on the Peltarion Platform.
          Retrieved from https://peltarion.com/knowledge-center/documentation/evaluation-view/classification-loss-metrics/micro-precision
        
        - MICRO-RECALL:
          Micro-recall on the Peltarion Platform. (2021). Micro-recall on the Peltarion Platform.
          Retrieved from https://peltarion.com/knowledge-center/documentation/evaluation-view/classification-loss-metrics/micro-recall
          
        - MICRO-F-SCORE:
          Adapted from Micro F1-score on the Peltarion Platform. (2021). Micro F1-score on the Peltarion Platform. 
          Retrieved from https://peltarion.com/knowledge-center/documentation/evaluation-view/classification-loss-metrics/micro-f1-score
  
    Inputs:
        - true_aspects (pandas Series): True aspects for each tweet
        - aspect_preds (pandas Series): Model's predicted aspects for each tweet
        - beta (float): The strength of recall versus precision in the F-score
        
    Outputs:
        - class metrics (dict): Dictionary of class-level metrics: false positive, true positive and precision
        - micro_precision (float): Micro-averaged precision
        - micro_recall (float): Micro-averaged recall
        - micro_fscore (float): Micro-averaged f-measure score
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
    weighted_precision, weighted_recall, weighted_precision
        
    #COMPUTE MICRO-AVERAGED PRECISION, RECALL AND F-score
    
    #Counters to track class aggregated metrics
    TP_sum, FP_sum, FN_sum = 0, 0, 0
     
    #Iterate through all the classes
    for aspect_key in class_metrics.keys():
        
        #Get the TP
        TP_sum += class_metrics[aspect_key]['TP']
        
        #Get the FP
        FP_sum += class_metrics[aspect_key]['FP']
        
        #Get the FN
        FN_sum += class_metrics[aspect_key]['FN']
        
    #Micro-precision
    micro_precision = TP_sum/(TP_sum + FP_sum)
    
    #Micro recall
    micro_recall = TP_sum/(TP_sum + FN_sum)
    
    #Micro F-Score
    micro_fscore = ((1 + beta**2) * micro_precision * micro_recall)/(beta**2 * micro_precision + micro_recall)
    
    #Return class level metrics, micro-precision, micro-recall and micro-f measure
    return class_metrics, micro_precision, micro_recall, micro_fscore


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
    