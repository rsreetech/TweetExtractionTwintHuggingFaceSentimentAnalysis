import twint
# Set up TWINT config
c = twint.Config()
c.Search = "Oneplus 9 pro"
# Custom output format
c.Limit = 3000
c.Pandas = True
twint.run.Search(c)

def column_names():
    return twint.output.panda.Tweets_df.columns
def twint_to_pd(columns):
    return twint.output.panda.Tweets_df[columns]

column_names()
tweet_df = twint_to_pd(["date", "username", "tweet", "hashtags", "likes_count"])
tweet_df.head(10)

print(len(tweet_df))

from transformers import pipeline
sentiment_classifier = pipeline('sentiment-analysis')

results = sentiment_classifier(tweet_df['tweet'].tolist())

for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
