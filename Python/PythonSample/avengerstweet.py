import tweepy
import pandas as pd
import os

consumer_key = ''
consumer_secret = ''
twitter_access_token = ''
twitter_access_secret = ''


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(twitter_access_token, twitter_access_secret)
twitter_api = tweepy.API(auth)

keyword = "어벤져스"
api_result = []
result = []



for i in range(1, 11):
    tweets = twitter_api.search(keyword)
    for tweet in tweets:
        result.append([tweet.id_str, tweet.user.name, tweet.created_at, tweet.text, tweet.user.followers_count, tweet.user.friends_count, tweet.favorite_count, tweet.retweet_count])



tweetdata = pd.DataFrame(result, columns=['id', 'name', 'created_at', 'text', 'followers_count', 'friends_count', 'favorite_count', 'retweet_count'])

if not os.path.exists('tweetavengers.csv'):
    tweetdata.to_csv('tweetavengers.csv', index=False, mode='w', encoding='utf-8-sig')
else:
    tweetdata.to_csv('tweetavengers.csv', index=False, mode='a', encoding='utf-8-sig', header=False)