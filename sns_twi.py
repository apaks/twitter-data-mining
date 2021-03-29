#author: apaks

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import snscrape.modules.twitter as sntwitter
import twidm_functions as twidm

# from bokeh.plotting import figure, ColumnDataSource, show
# from bokeh.models import HoverTool
import re
# download data
import yfinance as yf
#plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# sns.set_context("paper")
plt.rc("axes.spines", top=False, right=False)

def get_source(source):
    st_str = source.find(">") + 1
    end_str = source.find("</a>")
    return source[st_str:end_str]

@st.cache
def load_tweets(q_str, max_tweets = 100):

    # Creating list to append tweet data to
    tweets_list = []
    
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for idx, tweet in enumerate(sntwitter.TwitterSearchScraper(q_str).get_items()):
        if idx > max_tweets:
            break
        source = get_source(tweet.source)
        text = tweet.content.lower()
        # do not include bots and giveaways
        if "Twit" in source and "away" not in text and "give" not in text:
            tweets_list.append([tweet.date, tweet.id, text, tweet.user.username, tweet.replyCount, tweet.retweetCount,
                                tweet.likeCount, tweet.quoteCount,  source, tweet.url,
                tweet.user.id, tweet.user.description , tweet.user.followersCount, tweet.user.friendsCount
                                ])

    # Creating a dataframe from the tweets list above
    df_tweets = pd.DataFrame(tweets_list, columns=['created_at', 'id', 'text',  'username',
                                    'replyCount', 'retweetCount', 'likeCount', 'quoteCount', "source", "tweet_url",
                                    'user_id', "user_bio" , 'followers_count', 'friends_count' ])
    
    return df_tweets


   



models= ["vader", "txtblob"]


st.title('Twitter Trends')
st.markdown("### Go to Navigation panel after loading tweets")

sel_flag = st.radio("Are you interested in", ["stocks", "crypto"])
user_input = st.text_input("Put a ticker name eg. TSLA or BTC)")

dt_range = st.date_input("Enter dates of interest", [])

# if st.button('Load tweets'):
if user_input and len(dt_range)==2:
    
    q_str = f"{user_input} {sel_flag} since:{dt_range[0]} until:{dt_range[1]} lang:en"  
    st.write(q_str)
    max_tweets = 10000
    data = load_tweets(q_str, max_tweets)
    df_tweets = data.copy()
    st.write("{} Tweets are loaded".format(df_tweets.shape[0]))

    regex = re.compile('[^a-zA-Z]')
    #First parameter is the replacement, second parameter is your input string
    user_input = regex.sub('', user_input)
    if sel_flag == "stocks":
        # try:
        fin_data = yf.download(tickers = user_input, start = dt_range[0], end = dt_range[1], interval = '15m')
        fig = twidm.plot_ohlc(fin_data, user_input)
        st.plotly_chart(fig)
        # except:
        #     st.warning("Cannot download data, did you choose stocks?")
    else:
        # try:
        fin_data = yf.download(tickers = user_input + "-USD", start = dt_range[0], end = dt_range[1], interval = '30m')
        fig = twidm.plot_ohlc(fin_data, user_input)
        st.plotly_chart(fig)
        # except:
        #     st.warning("Cannot download data, did you choose crypto")



else:
    st.text("No input, type the ticker name")

st.sidebar.title("Navigation")



# page = st.sidebar.selectbox("Choose a page", ['Exploration', 'Tweets', 'Users'])
page = st.sidebar.radio("Choose a page", ('Load tweets', 'Sentiment across time', 'Sentiment across users'))

st.sidebar.title("About")
st.sidebar.warning("Let's see whether we can predict trends in stocks/crypto market from the sentiment analysis of tweets")
st.sidebar.info("- Retweets are not included            \n \
- Only recent tweets are loaded        \n \
- Seacrh limited to ~20K tweets        \n \
Powered by [Streamlit] (https://docs.streamlit.io/en/stable/api.html), \
    [Tweepy] (http://docs.tweepy.org/en/latest/), [yfinance](https://pypi.org/project/yfinance/)   \
    and [Plotly](https://plotly.com/)          \n " )

if page == 'Load tweets':
    
    if user_input and len(dt_range)==2:
        st.write(df_tweets.head())
        st.markdown('### Top 3 tweets')
        
        for idx in df_tweets.nlargest(3, "retweetCount").index:
            st.write(df_tweets.iloc[idx].text)
            st.write(df_tweets.iloc[idx].created_at)
            st.write(df_tweets.iloc[idx][["retweetCount", 'likeCount' , "followers_count"]])
            st.write(df_tweets.iloc[idx].tweet_url)
           
elif page == "Sentiment across time":
    st.title('Lets analyze tweets')
    # try:
    d_txtblob, d_vader, count_terms, count_hash = twidm.sentiment_analysis(df_tweets)
    df_tweets[ "txtblob"] = df_tweets["id"].map(d_txtblob)
    df_tweets["vader"] = df_tweets["id"].map(d_vader)
    # except:
    #     st.write("Load some tweets first")

    if st.checkbox("Activity across time"):
        # plot tweet counts over time
        fig = twidm.get_tweet_counts_overtime(fin_data, df_tweets["created_at"], user_input, "10T" )
        st.plotly_chart(fig)

    if st.checkbox("Tweet processing and sentiment analysis"):
  
        st.markdown('### The most common words')
        st.text(count_hash.most_common(100))
        st.text(count_terms.most_common(100))

        # plot wordcloud
        _input = count_terms
        fig = twidm.plot_wordcloud(_input)
        st.pyplot(fig)

        st.markdown('### Sentiment analysis')
  
        st.write(df_tweets.head())
        
        columns = st.multiselect(
            label='What column to you want to display', options= ["retweetCount", 
            "followers_count", "friends_count", "likeCount"])
        # col = "likeCount"
        if len(columns) > 0:
            for col in columns:
                st.write(col)
                fig = twidm.sentim_by_col(df_tweets, fin_data, col, models[0])
                st.plotly_chart(fig)

                fig = twidm.sentim_by_col(df_tweets, fin_data, col, models[1])
                st.plotly_chart(fig)

    if st.checkbox("Sentiment analysis across time"):
        
        tmp =  df_tweets[["created_at", "vader", "txtblob"]].set_index("created_at").resample('1H').mean().reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(x = fin_data.index, y=fin_data['High'],
                    name=user_input), secondary_y = False)

        fig.add_trace(go.Scatter(x=tmp['created_at'], y= tmp["vader"],
                    name='vader'), secondary_y = True)

        fig.add_trace(go.Scatter(x=tmp['created_at'], y= tmp["txtblob"],
                    name='txtblob'), secondary_y = True)

        st.plotly_chart(fig)


else:
    st.title('Lets look at users')
    # st.write(df_tweets.nlargest(5, ["followers_count", "retweetCount"]))
    st.markdown("### Users with trending tweets")
    try:

        st.write(df_tweets.nlargest(5, ["retweetCount", "followers_count"]))
    except: 
        st.write("load some tweets first")

    if st.checkbox("Cluster users on their profile description"):
        # print("")
        unique_user_map = dict(zip(df_tweets.user_id.values, df_tweets.user_bio.values))

        unique_user_cnt = len(unique_user_map.keys())
        vec = TfidfVectorizer(preprocessor=replace_urls,
                            tokenizer=my_tokenizer,
                            stop_words=stop,
                            max_features=unique_user_cnt//50,
                            )

        # we need to maintain the same ordering of users and bios
        unique_users = []
        unique_bios = []
        for user, bio in unique_user_map.items():
            unique_users.append(user)
            if bio is None:
                # special case for empty bios
                bio = ''
            unique_bios.append(bio)

        # calculate the data matrix
        bio_matrix = vec.fit_transform(unique_bios)
        st.text("Input matrix")
        st.write(pd.DataFrame(bio_matrix[200:210].todense(),
              columns=[x for x in vec.get_feature_names()]).iloc[:,0:15])
        
        # run hdbscan on user bios
        hdbs = twidm.fit_hdbscan(bio_matrix)
        fig = plot_hdbs_clustersize(hdbs)
        st.pyplot(fig)
        st.markdown("#### Strongest features")
        strongest_features(hdbs, vec, topk=15)

        # fit umap
        # embedding = fit_umap(bio_matrix.todense())

        # df_bios = get_plottable_df(unique_users, unique_bios, embedding, hdbs.labels_)
        # st.write(df_bios.head())

        # st.bokeh_chart(plot_umap(df_bios.sample(5000),
        #        'UMAP of clustered users ["(cluster #) bio"]'))
        
        if st.checkbox("Opinion mining across identified user groups"):
            d_txtblob, d_vader, count_terms, count_hash = twidm.sentiment_analysis(df_tweets)
            d_user_label = dict(zip(unique_users, hdbs.labels_))

            df_tweets[ "txtblob"] = df_tweets["id"].map(d_txtblob)
            df_tweets["vader"] = df_tweets["id"].map(d_vader)
            df_tweets["cluster"] = df_tweets["user_id"].map(d_user_label)

            st.write(df_tweets.head())
            st.markdown("### Sentiment by groups")
            fig, ax = plt.subplots(2, 1,  sharey = True, sharex = True)
            for idx, m in enumerate(models):  
                sns.violinplot(x = "cluster", y = m, data = df_tweets, ax = ax[idx])
            st.pyplot(fig)
        
            if st.checkbox("Sentiment analysis across time"):

                columns = st.multiselect( 
                label='What cluster to you want to display', options= sorted(df_tweets.cluster.unique().tolist()))
         
                fig, ax = plt.subplots(2, 1, sharex = True, sharey = True)
                if len(columns) > 0:
                    for col in columns:

                        tmp = df_tweets[df_tweets.cluster == col]
                        tmp =  tmp[["created_at", "cluster", models[0], models[1]]].set_index("created_at").resample('1H').mean().reset_index()
                        sns.lineplot(x = "created_at" , y = models[0],  data = tmp, ci = False , ax = ax[0], label = col )
                        sns.lineplot(x = "created_at" , y = models[1],  data = tmp, ci = False , ax = ax[1] )

                        date_form = DateFormatter("%m-%d")
                        ax[1].xaxis.set_major_formatter(date_form)
                        ax[1].set_xlabel('Date')
                        
                        # plt.legend(bbox_to_anchor=(.7, 1), loc=2, borderaxespad=0.)
                        plt.xticks(rotation = 45)
                        plt.tight_layout()
                    st.pyplot(fig)
           