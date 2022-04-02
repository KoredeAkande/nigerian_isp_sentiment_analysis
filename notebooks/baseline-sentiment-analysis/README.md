# Nigerian Internet Service Provider Sentiment Analysis [V1]


## 1. Twitter Data Extraction Jupyter Notebook

Code showing early attempts at extracting ISP data from Twitter API. This has been unsuccessful as of Sept 14th, 2021.

### Challenges so far:
- Inconsistencies in Twitter API documentation, resulting in usage of wrong parameters or wasted effort
- API version restricts the radius which you can map around a point, thus changing my current strategy
- Not all tweets return a coordinate, this motivates my decision to pivot from my current idea of analyzing ISPs by local government areas (LGAs)


## 2. Nigerian ISP Sentiment Analysis Jupyter Notebook

### TL,DR:
- Accounting for Nigerian Pidgin English (i.e. creole) is very important if we are to properly classify tweets as there were a lot of wrong classification seemingly as a result of the difference in grammatical structure

- It is important to pinpoint the target/subject of words in a tweet to avoid misattribution of positive/negative words.

- Connotations of certain sentences are positive/negative even though the sentence might appear objectively neutral e.g. You should get IPNX. 

### Objectives:
- Setup a working sentiment analysis pipeline
- Build familiarity with the Twitter API
- Diagnose issues with publicly available sentiment analysizers when working with Nigerian data.

### Requirements:
- Tweepy (pip install tweepy)
- Textblob (pip install Textblob) for Sentiment Analysis
- Twitter developer account
  - Create a developer account
  - Start a project
  - Get keys and access tokens (these will be inputted in the notebook)

### Process:

**API Data Extraction**\
After connecting to the Twitter API, I retrieved about 15 tweets per ISP for two ISPs (IPNX and Spectranet) near Lagos, Nigeria. One was retrieved through the typical search function, which allows you retrieve tweets no later than a week ago. Another was retrieved through the full archive search, which allows retrieval of tweets as late as 2006. Certain tweet properties e.g. the date, text, location, etc. were then extracted from the tweet object and saved as rows in a pandas DataFrame.

**Data Cleaning**\
The tweets were cleaned for hashtags, @ symbols, links, etc.


**Sentiment Analysis**\
The polarity of the different tweets were found using Textblob. Subsequently, the tweets' sentiments were also determined manually. The two sentiments were then compared for matches. This helped in diagnosing weaknesses of the Textblob sentiment analyzer and important areas to improve upon when analyzing the sentiment of Nigerian ISP data.

### Conclusions
- Accounting for Pidgin English and slangs is very important if we are to properly classify tweets.

- It is important to pinpoint the target/subject of words in a tweet to avoid misattribution of positive/negative words.

- Connotations of certain sentences are positive/negative even though the sentence might appear objectively neutral e.g. I want IPNX! 

