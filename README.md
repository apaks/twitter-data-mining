# Twitter Trends  
How things get popular and viral on social media? While influencers and large companies mostly shape trends on social media, why some products are more visible than others? I built an app that utilizes Twitter data to gain insight into how public opinion on a specific issue/product/service is shaped by and distributed across different user groups and time. Sentiment analysis was performed using nltk and textblob and the results show how public opinion might be differentially distributed across different user groups and time. See demo below.  

![TwiTr app Demo](https://github.com/apaks/twitter-data-mining/blob/master/demo.gif)

**Fetch tweets using _Tweepy_**    

**1. Save them in:**  
  - DF 
  - PostgreSQL DB
  - .json

**2. Analyze and visualize Twitter data using varous libraries:**  
  - *nltk*
  - *sklearn*  
  - *wordcloud*
  - *NetworkX*
  - tSNE/UMAP

**3. Perform:**  
  - basic text processing  
  - common term/hashtag/n-gram extraction  
  - time-series
  - user clustering based on their bio
    - k-means
    - HDBSCAN
  - construct graph based on user interactions  
  - community detection, graph visualization [Gephi](https://gephi.org/)
  

