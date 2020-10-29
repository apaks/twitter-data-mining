# Twitter Trends  
> How things get popular and viral on social media? While influencers and big companies mostly shape social media, why some products are more visible than others? I plan to build a web tool that utilizes Twitter data to gain insight into how public opinion on a specific issue/product/service is shaped by and distributed across different user groups and time.  We perform sentiment analysis (nltk, textblob, flair) on Tweets and visualize how it is distributed across identified groups and time. 

![TwiTr app Demo](https://github.com/apaks/twitter-data-mining/blob/master/demo.gif)

**Fetch tweets using _Tweepy_**    

**1. Save them in:**  
  - DF
  - PostgreSQL DB
  - .json

**2. Process and analyze tweets using:**  
  - *nltk*
  - *wordcloud*
  - *sklearn*  
  - *NetworkX*
  - tSNE

**3. Perform:**  
  - basic text processing  
  - common term/hashtag/n-gram extraction  
  - time-series
  - user clustering based on their bio
    - k-means
    - HDBSCAN
  - construct graph based on user interactions  
  - community detection, graph visualization [Gephi](https://gephi.org/)
  

