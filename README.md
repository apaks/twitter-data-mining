# Twitter Trends  
How things get popular and viral on social media? While influencers and big companies mostly shape trends on social media, why some products are more visible than others? I plan to build a web tool that utilizes Twitter data to gain insight into how public opinion on a specific issue/product/service is shaped by and distributed across different user groups and time. I performed sentiment analysis using nltk, textblob, and flair on tweets and visualized how public opinion was distributed across identified user groups and evolved in time. 

![TwiTr app Demo](https://github.com/apaks/twitter-data-mining/blob/master/demo.gif)

**Fetch tweets using _Tweepy_**    

**1. Save them in:**  
  - DF
  - PostgreSQL DB
  - .json

**2. Analyze tw arious libraries:**  
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
  

