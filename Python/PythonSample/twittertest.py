import tweepy
import pandas as pd
import os

consumer_key = 'Mwo8KsPPsUH2a2RAuBAGqhOtD'
consumer_secret = 'uwaZjK3nNCZjUbkUiMRLNsrYH9P0yo6zt2VbE2Ctgk08QtGqZU'
twitter_access_token = '1311709615923445760-Khswk7BPmSD4GFUDdp2EGlGAZJAjIv'
twitter_access_secret = 'DD1rtTErJUOfVGjfjsL5ahDKm0ENy9IVxM4BLno0Cshoa'

#twitter_api = twitter.Api(consumer_key=twitter_consumer_key, consumer_secret=twitter_consumer_secret, access_token_key=twitter_access_token, access_token_secret=twitter_consumer_secret)
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(twitter_access_token, twitter_access_secret)
twitter_api = tweepy.API(auth)

keyword = "영화"
api_result = []
result = []

#tweet = twitter_api.search(keyword)

# for i in range(1, 3):
#     tweets = twitter_api.search(keyword)
#     for tweet in tweets:
#         result.append(tweet)

# print(len(result))
# print(result[0])

# for tw in tweet:
#     api_result.append(tw.text)

# #print(api_result)
# for aaa in api_result:
#     print(aaa)

for i in range(1, 11):
    tweets = twitter_api.search(keyword)
    for tweet in tweets:
        result.append([tweet.id_str, tweet.user.name, tweet.created_at, tweet.text, tweet.user.followers_count, tweet.user.friends_count, tweet.favorite_count, tweet.retweet_count])

#print(len(result))
#print(result)

# for i in range(1, 3):
#     tweets = twitter_api.search(keyword)
#     for tweet in tweets:
#         result.append(tweet.user.followers_count)

# print(len(result))
# print(result[0])

tweetdata = pd.DataFrame(result, columns=['id', 'name', 'created_at', 'text', 'followers_count', 'friends_count', 'favorite_count', 'retweet_count'])

if not os.path.exists('tweetdata.csv'):
    tweetdata.to_csv('tweetdata.csv', index=False, mode='w', encoding='utf-8-sig')
else:
    tweetdata.to_csv('tweetdata.csv', index=False, mode='a', encoding='utf-8-sig', header=False)