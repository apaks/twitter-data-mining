#@author apaks

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
import operator 
import string
from collections import Counter

from wordcloud import WordCloud
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from textblob import TextBlob
# cluster users

import hdbscan

sian = SentimentIntensityAnalyzer()

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



def get_tweet_counts_overtime(fin_data, ts, user_input, res_window = "1H" ):
    
    tmp = ts
    ones = [1]*len(tmp)
    tmp = pd.DataFrame(ones, index = tmp)
    tmp = tmp.resample(res_window).sum().fillna(0).reset_index()
    tmp.columns = ['date', 'freq']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=fin_data.index, y=fin_data['High'],
            ), secondary_y = False)

    fig.add_trace(go.Scatter(x=tmp["date"], y=tmp["freq"], 
            name = "count of tweets"), secondary_y = True)

    fig.update_layout(
        title=user_input.upper(),
        plot_bgcolor="#FFFFFF",
        hovermode="x",
        hoverdistance=100, # Distance to show hover label of data point
        spikedistance=-1, # Distance to show spike
        xaxis=dict(
    #         title="time",
            linecolor="#BCCCDC",
    #         showspikes=True, # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="toaxis+across",
        ),
        yaxis=dict(
            title="Price",
            linecolor="#BCCCDC"
        ),
        yaxis2=dict(
            title="Count of tweets",
            linecolor="#BCCCDC"
        )
    )
    return fig

def plot_ohlc(fin_data, user_input):

    fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2],shared_xaxes=True,
                        vertical_spacing=0.02)
    fig.add_trace(go.Ohlc(x=fin_data.index,
            open=fin_data['Open'],
            high=fin_data['High'],
            low=fin_data['Low'],
            close=fin_data['Close']), row = 1, col = 1)

    fig.add_trace(go.Bar(x=fin_data.index, y=fin_data["Volume"], marker_color = "black" ), row = 2, col = 1)    
    fig.update(layout_xaxis_rangeslider_visible=False)

    fig.update_layout(
        title=user_input.upper(),
        plot_bgcolor="#FFFFFF",
        hovermode="x",
        hoverdistance=100, # Distance to show hover label of data point
        spikedistance=-1, # Distance to show spike
        xaxis=dict(
    #         title="time",
            linecolor="#BCCCDC",
    #         showspikes=True, # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="toaxis+across",
        ),
        yaxis=dict(
            title="Price",
            linecolor="#BCCCDC"
        ),
        yaxis2=dict(
            title="Volume",
            linecolor="#BCCCDC"
        )
    )
    return fig


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


def sentim_by_col(df_tweets, fin_data, col, y):
    if "fol" in col or "fri" in col:
        bins = [-1, 5000, 10000, 50000, 100000, 2e8]
        bin_labels = ["<5,000", "<10,000", "<50,000", "less than 100,000", "more than 100,000"]
    else:
        bins = [-1, 50, 100, 500, 1000, 2e7]
        bin_labels = ["<50", "<100", "<500", "less than 1,000", "more than 1,000"]
    tmp = pd.cut(df_tweets[col], bins = bins,  labels= bin_labels) 
    df_tweets[col + "_cut"] = tmp
    
    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = px.scatter(df_tweets, x="created_at", y = y, size= col, 
                hover_name="username",
                    color= col + '_cut',
                    log_x=False, size_max= 40)
    fig2 = px.line(x=fin_data.index, y=fin_data['High'])
    for tr in fig.data:
        subfig.add_trace(tr, secondary_y = True)
    for tr in fig2.data:
        subfig.add_trace(tr, secondary_y = False)
    subfig['layout']["yaxis"]["title"] = f"Sentiment algo: {y}"
    subfig['layout']["legend"]["title"] = col

    return subfig

@st.cache(show_spinner=False)
def sentiment_analysis(df_tweets):
    tweet_tokenizer = TweetTokenizer()
    d_txtblob = {}
    d_vader = {}

    count_terms = Counter()
    count_hash = Counter()
    # count terms and perform sentiment analysis
    for idx in df_tweets.index[:]:
        text = df_tweets.iloc[idx].text.lower()
        if len(text) < 5:
            continue
        cleaned_text = process_tweet(text)
        tweet_id = df_tweets.iloc[idx].id 
        
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
    tokens = [word for word in tokens if len(word) > 2 and "#" not in word]
    return tokens

@st.cache(show_spinner=False)
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

def strongest_features(model, vectorizer, bio_matrix, topk=10):
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
        matching_rows = np.where(model.labels_ == this_label)[0]
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

