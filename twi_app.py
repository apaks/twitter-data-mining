import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tweepy
import seaborn as sns
from wordcloud import WordCloud
from matplotlib.dates import DateFormatter

from nltk.tokenize import TweetTokenizer
import re
import operator 
from collections import Counter
from nltk.corpus import stopwords
import string

# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sian = SentimentIntensityAnalyzer()
from textblob import TextBlob

# cluster users
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
import umap
from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.models import HoverTool
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sns.set_context("paper")
plt.rc("axes.spines", top=False, right=False)


class TweetMiner(object):
    import auth_twitter
    # number of tweets per one pull; there is limit on this
    result_limit    =   20    
    data            =   []
    api             =   False
    
    twitter_keys = {
        'consumer_key':        auth_twitter.consumer_key,
        'consumer_secret':     auth_twitter.consumer_secret,
        'access_token_key':    auth_twitter.access_token,
        'access_token_secret': auth_twitter.access_token_secret
    }
    
    
    def __init__(self, keys_dict=twitter_keys, api=api, result_limit = 100):
        
        self.twitter_keys = keys_dict
        
        auth = tweepy.OAuthHandler(keys_dict['consumer_key'], keys_dict['consumer_secret'])
        auth.set_access_token(keys_dict['access_token_key'], keys_dict['access_token_secret'])
        
        self.api = tweepy.API(auth)
        self.twitter_keys = keys_dict
        
        self.result_limit = result_limit
    # @st.cache
    def mine_tweets_user(self, user="", last_tweet_id  =  False, max_pages=17):
        # keep track of last tweet id
        # multiply by the # of result_limit = total tweets
        data_page = []
        page =  1
        
        while page <= max_pages:
            if last_tweet_id:
                statuses   =   self.api.user_timeline(screen_name = user,
                                                     count = self.result_limit,
                                                     # get tweets older than last retrieved ones  
                                                     max_id = last_tweet_id - 1,
                                                     tweet_mode = 'extended',
                                                    )        
            else:
                statuses   =   self.api.user_timeline(screen_name=user,
                                                        count = self.result_limit,
                                                        tweet_mode = 'extended',
                                                        )
            for st in statuses:
                data_page.append(st._json)
                last_tweet_id = st.id

            page += 1
        # returns list of dict
        return data_page, last_tweet_id
    # @st.cache
    def mine_tweets_keyword(self, query = "", language = 'en', last_tweet_id  =  False,
                         max_pages=17):

        data_page = []
        page = 1
        
        while page <= max_pages:
            if last_tweet_id:
                statuses   =   self.api.search(q = query, lang = language,
                                                     count = self.result_limit,
                                                     # get tweets older than last retrieved ones  
                                                     max_id = last_tweet_id - 1,
                                                     tweet_mode = 'extended',
                                                    )        
            else:
                statuses   =   self.api.search(q = query, lang = language,
                                                        count = self.result_limit,
                                                        tweet_mode = 'extended',
                                                 )
    
            for st in statuses:
                data_page.append(st._json)
                last_tweet_id = st.id
            page += 1
        # returns list of dict
        return data_page, last_tweet_id

def get_status_text(df_row):
    #retweet
    try:
        if isinstance(df_row.retweeted_status, dict):
            text = df_row.retweeted_status['full_text']
        #quote
        elif isinstance(df_row.quoted_status, dict):
            text = df_row.quoted_status["full_text"]
        # regular tweet
    except:
        text = df_row['full_text']
    return text

def get_tweet_counts_overtime(ts, res_window = "1H"):
    
    tmp = ts
    ones = [1]*len(tmp)
    tmp = pd.DataFrame(ones, index = tmp)
    tmp = tmp.resample(res_window).sum().fillna(0).reset_index()
    tmp.columns = ['date', 'freq']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=fin_data.index, y=fin_data['High'],
            name=user_input), secondary_y = False)

    fig.add_trace(go.Scatter(x=tmp["date"], y=tmp["freq"], 
            name = "count of tweets"), secondary_y = True)

    return fig

@st.cache
def load_tweets(miner, q_word):
    counter = 1
    ls_master = []
    last_id = False
    while counter < 10:
        # print (counter)
        try:
            ls_tweets, last_tweet_id = miner.mine_tweets_keyword(query=q_word, language = 'en', 
                                        last_tweet_id = last_id, max_pages = 20)
            last_id = last_tweet_id
            ls_master.extend(ls_tweets)
        except:
            st.error("Limit is reached. Wait for 15 min before you can access Twitter API")
    #         time.sleep(16*60) #15 minute sleep time
            break
        if len(ls_tweets) < 1:
            break
        counter+=1
    df_tweets = pd.DataFrame(ls_master)
    return df_tweets


def select_columns(df_tweets): 
    # select relevant info
    df_tweets_2 = df_tweets[["created_at", "id", "retweet_count", "favorite_count"]]
    tmp = pd.DatetimeIndex(df_tweets_2["created_at"])
    df_tweets_2["created_at"] = tmp
    tmp = df_tweets["user"].apply(lambda x: x["followers_count"])
    df_tweets_2["followers_count"] = tmp
    tmp = df_tweets["user"].apply(lambda x: x["friends_count"])
    df_tweets_2["friends_count"] = tmp
    tmp = df_tweets["user"].apply(lambda x: x["id"])
    df_tweets_2["user_id"] = tmp
    tmp = df_tweets["user"].apply(lambda x: x["description"])
    df_tweets_2["user_bio"] = tmp
    tmp = df_tweets.apply(get_status_text, axis = 1)
    df_tweets_2["text"] = tmp
    # df_tweets_2 = df_tweets_2.drop_duplicates(subset = ["text"]).reset_index(drop = True)
    return df_tweets_2

def plot_wordcloud(_input):
    wc = WordCloud( contour_width=3, contour_color= 'steelblue',
                    background_color ='white', max_font_size=50, 
                    max_words=200, random_state=42, 
                    min_font_size = 10).generate_from_frequencies(_input) 
    
    # plot the WordCloud image     
    fig, ax = plt.subplots(figsize = (8, 8), facecolor = None)                   
    ax.imshow(wc, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    return fig


def sentim_by_col(col, y):
    if "fol" in col or "fri" in col:
        bins = [-1, 5000, 10000, 50000, 100000, 2e8]
        bin_labels = ["<5000", "<10000", "<50000", "less than 100000", "more than 100000"]
    else:
        bins = [-1, 50, 100, 500, 1000, 2e7]
        bin_labels = ["<50", "<100", "<500", "less than 1000", "more than 1000"]
    tmp = pd.cut(df_tweets_2[col], bins = bins,  labels= bin_labels) 
    df_tweets_2[col + "_cut"] = tmp
    
    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = px.scatter(df_tweets_2, x="created_at", y= y, size= col, 
                hover_name="user_bio",
                    color= col + '_cut',
                    log_x=False, size_max=40)
    fig2 = px.line(x=fin_data.index, y=fin_data['High'])
    for tr in fig.data:
        subfig.add_trace(tr, secondary_y = True)
    for tr in fig2.data:
        subfig.add_trace(tr, secondary_y = False)
    subfig['layout']["yaxis"]["title"] = f"Sentiment algo: {y}"
    subfig['layout']["legend"]["title"] = col

    return subfig

@st.cache
def sentiment_analysis(df_tweets_2):
    tweet_tokenizer = TweetTokenizer()
    d_txtblob = {}
    d_vader = {}

    count_terms = Counter()
    count_hash = Counter()
    # count terms and perform sentiment analysis
    for idx in df_tweets_2.index[:]:
        text = df_tweets_2.iloc[idx].text.lower()
        if len(text) < 5:
            continue
        cleaned_text = process_tweet(text)
        tweet_id = df_tweets_2.iloc[idx].id 
        
        # word freqeuncy
        terms_hash = [term for term in tweet_tokenizer.tokenize(text) 
                if term not in stop and len(term) > 3 and term.startswith('#')]
        terms_only = [term for term in tweet_tokenizer.tokenize(text) 
                if term not in stop and len(term) > 3 and
                not term.startswith(('#', '@'))] 
        
        # Update the counter
        count_hash.update(terms_hash)
        count_terms.update(terms_only)
        
        # sentiment analysis
        try:
            compound = sian.polarity_scores(cleaned_text)["compound"]
            d_vader[tweet_id] = round(compound, 5)

            txtblob = TextBlob(cleaned_text).sentiment[0]
            d_txtblob[tweet_id] = round(txtblob, 5)
        
        except Exception as e: 
            print (str(e))

    return d_txtblob, d_vader, count_terms, count_hash



models= ["vader", "txtblob"]
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation 
emoticons_str = r"""
(?:
    [:=;] # Eyes
    [oO\-]? # Nose (optional)
    [D\)\]\(\]/\\OpP] # Mouth
)"""
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'\n',
    r"^\s+|\s+$"
]
cleaner_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
def process_tweet(tweet): 
    return cleaner_re.sub("", tweet)



def replace_urls(in_string, replacement=None):
    """Replace URLs in strings. See also: ``bit.ly/PyURLre``

    Args:
        in_string (str): string to filter
        replacement (str or None): replacment text. defaults to '<-URL->'

    Returns:
        str
    """
    replacement = '<-URL->' if replacement is None else replacement
    pattern = re.compile('(https?://)?(\w*[.]\w+)+([/?=&]+\w+)*')
    return re.sub(pattern, replacement, in_string)

def my_tokenizer(in_string):
    """
    Convert `in_string` of text to a list of tokens using NLTK's TweetTokenizer
    """
    # reasonable, but adjustable tokenizer settings
    tokenizer = TweetTokenizer(preserve_case=False,
                               reduce_len=True,
                               strip_handles=True)
    tokens = tokenizer.tokenize(in_string)
    tokens = [word for word in tokens if len(word) > 2]
    return tokens

@st.cache
def fit_hdbscan(data):

    hdbs = hdbscan.HDBSCAN(min_cluster_size=100,
                        prediction_data=True,
                        core_dist_n_jobs=-1,
                        memory='data')
    hdbs.fit(data.todense())
    return hdbs

def plot_hdbs_clustersize(hdbs):
    # get the population sizes
    label_counts = Counter(hdbs.labels_)
    xs, ys = [], []
    for k,v in label_counts.items():
        xs.append(k)
        ys.append(v)
    fig, ax = plt.subplots()
    # draw the chart
    ax.bar(xs, ys)

    plt.xticks(range(-1, len(label_counts)))
    plt.ylabel('population')
    plt.xlabel('cluster label')
    plt.title('population sizes ({} clusters found by hdbscan)'.format(len(label_counts) - 1));
    return fig

def strongest_features(model, vectorizer, topk=10):
    """
    Helper function to display a simple text representation of the top-k most
    important features in our fit model and vectorizer.

    model: sklearn model
    vectorizer: sklearn vectorizer
    topk: k numbers of words to get per cluster

    """
    # these parts are model-independent

    features = vectorizer.get_feature_names()
    relevant_labels = [ x for x in set(model.labels_) if x >= 0 ]
    # -1 is a noise cluster
    for this_label in relevant_labels:
        matching_rows = np.where(hdbs.labels_ == this_label)[0]
        coeff_sums = np.sum(bio_matrix[matching_rows], axis=0).A1
        sorted_coeff_idxs = np.argsort(coeff_sums)[::-1]
        st.text('Cluster {}: '.format(this_label))
        tmp = " "
        for idx in sorted_coeff_idxs[:topk]:
            tmp = tmp + features[idx] + " "
        st.text(tmp)
        # st.text("")

def get_plottable_df(users, bios, two_d_coords, labels):
    """
    Combine the necessary pieces of data to create a data structure that plays
    nicely with the our 2d tsne chart.

    Note: assumes that all argument data series
    are in the same order e.g. the first user, bio, coords, and label
    all correspond to the same user.
    """
    # set up color palette
    num_labels = len(set(labels))
    colors = sns.color_palette('hls', num_labels).as_hex()
    color_lookup = {v:k for k,v in zip(colors, set(labels))}
    # combine data into a single df
    df = pd.DataFrame({'uid': users,
                       'text': bios,
                       'label': labels,
                       'x_val': two_d_coords[:,0],
                       'y_val': two_d_coords[:,1],
                      })
    # convert labels to colors
    df['color'] = list(map(lambda x: color_lookup[x], labels))
    return df

@st.cache
def fit_umap(data):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    return embedding

def plot_umap(df, title='Umap plot'):
    # add our DataFrame as a ColumnDataSource for Bokeh
    plot_data = ColumnDataSource(df)
    # configure the chart
    umap_plot = figure(title=title, plot_width=800, plot_height=700, tools=('pan, box_zoom, reset'))
    # add a hover tool to display words on roll-over
    umap_plot.add_tools(
        HoverTool(tooltips = """<div style="width: 400px;">(@label) @text</div>""")
    )
    # draw the words as circles on the plot
    umap_plot.circle('x_val', 'y_val',
                     source=plot_data,
                     color='color',
                     line_alpha=0.2,
                     fill_alpha=0.1,
                     size=7,
                     hover_line_color='black')
    # configure visual elements of the plot
    umap_plot.title.text_font_size = '12pt'
    umap_plot.xaxis.visible = False
    umap_plot.yaxis.visible = False
    umap_plot.grid.grid_line_color = None
    umap_plot.outline_line_color = None
    return umap_plot




st.title('Twitter Trends')
st.markdown("### Go to Navigation panel after loading tweets")

sel_flag = st.radio("Are you interested in", ["stocks", "crypto"])
user_input = st.text_input("Put a ticker name eg. TSLA or BTC)")

# if st.button('Load tweets'):
if user_input:
    
    q_word = user_input + " -filter:retweets"
    miner = TweetMiner(result_limit = 100)
    df_tweets = load_tweets(miner, q_word)

    st.write("{} Tweets are loaded".format(df_tweets.shape[0]))

    df_tweets_2 = select_columns(df_tweets)
    regex = re.compile('[^a-zA-Z]')
    #First parameter is the replacement, second parameter is your input string
    user_input = regex.sub('', user_input)
    if sel_flag == "stocks":
        fin_data = yf.download(tickers = user_input, period = '7d', interval = '15m')
        fig = px.line(fin_data, x= fin_data.index, y= 'High', title = f"7 day history of {user_input}")
        st.plotly_chart(fig)
    else:
        fin_data = yf.download(tickers = user_input + "-USD", period = '7d', interval = '30m')
        fig = px.line(fin_data, x= fin_data.index, y= 'High', title = f"7 day history of {user_input}")
        st.plotly_chart(fig)

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
    
    if user_input:
        st.write(df_tweets_2.head())
        st.markdown('### Top 3 tweets')
        
        for idx in df_tweets_2.nlargest(3, "retweet_count").index:
            st.write(df_tweets_2.iloc[idx].text)
            # st.write(df_tweets_2.iloc[idx].created_at)
            st.write(df_tweets_2.iloc[idx][["retweet_count", 'favorite_count' , "followers_count"]])

           
elif page == "Sentiment across time":
    st.title('Lets analyze tweets')
    try:
        d_txtblob, d_vader, count_terms, count_hash = sentiment_analysis(df_tweets_2)
        df_tweets_2[ "txtblob"] = df_tweets_2["id"].map(d_txtblob)
        df_tweets_2["vader"] = df_tweets_2["id"].map(d_vader)
    except:
        st.write("Load some tweets first")

    if st.checkbox("Activity across time"):
        # plot tweet counts over time
        fig = get_tweet_counts_overtime(df_tweets_2["created_at"], "10T")
        st.plotly_chart(fig)

    if st.checkbox("Tweet processing and sentiment analysis"):
  
        st.markdown('### The most common words')
        st.text(count_hash.most_common(100))
        st.text(count_terms.most_common(100))

        # plot wordcloud
        _input = count_terms
        fig = plot_wordcloud(_input)
        st.pyplot(fig)

        st.markdown('### Sentiment analysis')
  
        st.write(df_tweets_2.head())
        
        columns = st.multiselect(
            label='What column to you want to display', options= ["retweet_count", 
            "followers_count", "friends_count", "favorite_count"])
        # col = "favorite_count"
        if len(columns) > 0:
            for col in columns:
                st.write(col)
                fig = sentim_by_col(col, models[0])
                st.plotly_chart(fig)

                fig = sentim_by_col(col, models[1])
                st.plotly_chart(fig)

    if st.checkbox("Sentiment analysis across time"):
        
        tmp =  df_tweets_2[["created_at", "vader", "txtblob"]].set_index("created_at").resample('1H').mean().reset_index()
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
    # st.write(df_tweets_2.nlargest(5, ["followers_count", "retweet_count"]))
    st.markdown("### Users with trending tweets")
    try:

        st.write(df_tweets_2.nlargest(5, ["retweet_count", "followers_count"]))
    except: 
        st.write("load some tweets first")

    if st.checkbox("Cluster users on their profile description"):
        # print("")
        unique_user_map = dict(zip(df_tweets_2.user_id.values, df_tweets_2.user_bio.values))

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
        hdbs = fit_hdbscan(bio_matrix)
        fig = plot_hdbs_clustersize(hdbs)
        st.pyplot(fig)
        st.markdown("#### Strongest features")
        strongest_features(hdbs, vec, topk=15)

        # fit umap
        embedding = fit_umap(bio_matrix.todense())

        df_bios = get_plottable_df(unique_users, unique_bios, embedding, hdbs.labels_)
        st.write(df_bios.head())

        st.bokeh_chart(plot_umap(df_bios.sample(5000),
               'UMAP of clustered users ["(cluster #) bio"]'))
        
        if st.checkbox("Opinion mining across identified user groups"):
            d_txtblob, d_vader, count_terms, count_hash = sentiment_analysis(df_tweets_2)
            d_user_label = dict(zip(df_bios.uid, df_bios.label))

            df_tweets_2[ "txtblob"] = df_tweets_2["id"].map(d_txtblob)
            df_tweets_2["vader"] = df_tweets_2["id"].map(d_vader)
            df_tweets_2["cluster"] = df_tweets_2["user_id"].map(d_user_label)

            st.write(df_tweets_2.head())
            st.markdown("### Sentiment by groups")
            fig, ax = plt.subplots(2, 1,  sharey = True, sharex = True)
            for idx, m in enumerate(models):  
                sns.violinplot(x = "cluster", y = m, data = df_tweets_2, ax = ax[idx])
            st.pyplot(fig)
        
            if st.checkbox("Sentiment analysis across time"):

                columns = st.multiselect( 
                label='What cluster to you want to display', options= sorted(df_tweets_2.cluster.unique().tolist()))
         
                fig, ax = plt.subplots(2, 1, sharex = True, sharey = True)
                if len(columns) > 0:
                    for col in columns:

                        tmp = df_tweets_2[df_tweets_2.cluster == col]
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
           