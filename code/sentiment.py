import pickle
import requests
import time
import pandas as pd

model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
hf_token = "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"

API_URL = "https://api-inference.huggingface.co/models/" + model
headers = {"Authorization": "Bearer %s" % (hf_token)}

def analysis(data):
    payload = dict(inputs=data, options=dict(wait_for_model=True))
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


df = pickle.load(open('allTweetsDF.p','rb'))
tweets = list(df['text'])

tweets_analysis = []
for tweet in tweets:
    try:
        sentiment_result = analysis(tweet)[0]
        top_sentiment = max(sentiment_result, key=lambda x: x['score']) # Get the sentiment with the higher score
        tweets_analysis.append({'tweet': tweet, 'sentiment': top_sentiment['label']})
 
    except Exception as e:
        print(e)


# Load the data in a dataframe
pd.set_option('max_colwidth', None)
pd.set_option('display.width', 3000) 
df = pd.DataFrame(tweets_analysis)

# Show a tweet for each sentiment 
display(df[df["sentiment"] == 'Positive'].head(1))
display(df[df["sentiment"] == 'Neutral'].head(1))
display(df[df["sentiment"] == 'Negative'].head(1))