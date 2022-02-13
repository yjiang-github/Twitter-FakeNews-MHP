# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:26:16 2021

@author: yjian
"""


"""

this file is for testing the new Twitter APIA v2 with new token&key in academic research project

with access to full archive

"""
#%%
"""
import packages
"""
import json
import os
import tweepy
import re
import time
from tqdm import tqdm
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
from pytz import timezone
import logging
import OpenSSL
import datetime
from twarc.client2 import Twarc2
from twarc.expansions import flatten
import math
import csv

#%%
createVar = locals()

#%%
"""
--- collecting tweets associated with fake news of "obama was arrested for espionage ---
"""

#%%
"""
--- key & token ---

"""

consumer_key = '6US9ar8fUELd82s3QyXCvJtou'  
consumer_secret = 'MARHHWlMKbmKn5qRKeRpz936g40jEk6iTGxMQmTZc6vDumGvHw'  
access_token = '4876272161-1DbKoux957KZA3TXJCGCvmZgNxDDZmL6izPvIeP'  
access_token_secret = 'musqMaYQfvDk8ppEtYe35OFjUS7ETJ2WOwJlqhmAQgtn9'  
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAB1sWQEAAAAABurcSFKai%2BZEAjHH7RADmURqevY%3D15fOlvJhJYm7L5q0aZlc09i2VqhUvZjaHbU9NzjjC1Xi5GLdFA'





"""
--- API ---
​
"""


t = Twarc2(bearer_token=bearer_token)


#%%

keyword01 = 'obama arrested espionage'
keyword02 = 'obama arrested fact check'
keyword03 = 'obama arrested factcheck'
url01 = 'url:'+'www.conservativebeaver.com/2020/11/28/former-president-barack-obama-arrested-for-espionage'
url02 = 'url:'+'www.politifact.com/factchecks/2020/dec/03/blog-posting/no-obama-wasnt-arrested-espionage/'
url03 = 'url:'+'www.snopes.com/fact-check/obama-wasnt-arrested/'

#%%

dict_keyword01 = {}
dict_keyword02 = {}
dict_keyword03 = {}
dict_url01 = {}
dict_url02 = {}
dict_url03 = {}
dict_urls = {}

dict_temp = {}

dict_content = {} # dictionary for tweet content, incluing all tweets in conversations
dict_structure = {} # dictionary for conversation structure
dict_missing = {} # dictionary for missing tweets' ids

#%%
"""
--- search tweets ---

"""
start_time = '2020-11-27T00:00:01Z'
end_time = '2021-01-31T23:59:59Z'




#%%
""" twarc2 """
tweets = t.search_all(query=url03, start_time=start_time, end_time=end_time, max_results=100)


for search in tweets:
    dict_temp = search
    for tweet in search['data']:
        if tweet['lang'] == 'en':
            dict_url03[tweet['id']] = tweet
        

#%%
""" combine all dictionaries """
dict_combined = dict_keyword01.copy()

#%%
for tweet_id in dict_keyword03.keys():
    if tweet_id not in dict_combined.keys():
        dict_combined[tweet_id] = dict_keyword03[tweet_id]


#%%
""" collect more relevant urls from fake news tweets """
dict_temp = {}

list_link = []

url01s = 'conservativebeaver.com/2020/11/28/for'
url02s = 'politifact.com/factchecks/2020/dec/03/blog-posting/no-obama-wasnt'
url03s = 'snopes.com/fact-check/obama-wasnt-arrested'

for tweet_id in dict_combined.keys():
    if 'entities' in dict_combined[tweet_id] and 'urls' in dict_combined[tweet_id]['entities'].keys():
        for url in dict_combined[tweet_id]['entities']['urls']:
            if url01s not in url['expanded_url'] and url02s not in url['expanded_url'] and url03s not in url['expanded_url'] and\
            url['expanded_url'] not in dict_temp.values() and 'photo' not in url['expanded_url']:
                dict_temp[tweet_id] = url['expanded_url']

list_temp = list(dict_temp.values())

#%%
""" filter the links """
for link in list_temp:
    if 'obama' in link and 'arrest' in link:
        list_link.append(link)

for link in list_link:
    list_temp.remove(link)

#%%
""" mannuly checkout the links """
i = 0
while i <= 30:
    list_link.append(list_temp[0])
    list_temp.pop(0)
    i += 1
    
#%%

""" collect tweets associated with filtered links """

for link in tqdm(list_link):    
    if 'https://' in link:
        temp_link = link[8:-1]
    elif 'http://' in link:
        temp_link = link[7:-1]
    else:
        temp_link = link
        
    url = 'url:'+temp_link
    tweets = t.search_all(query=url, start_time=start_time, end_time=end_time, max_results=100)
    for search in tweets:
        dict_temp= search['data']
        for tweet in search['data']:
            if tweet['lang'] == 'en' and tweet['conversation_id'] not in dict_urls.keys():
                dict_urls[tweet['conversation_id']] = tweet
    
#%%
""" save tweets into dict_combined """
for tweet_id in dict_urls.keys():
    if tweet_id not in dict_combined.keys():
        dict_combined[tweet_id] = dict_urls[tweet_id]

#%%

""" collect more tweets regarding the public metrics (descendants) """

list_conversationid = list(dict_combined.keys())
# get the original tweets by conversation_ids
for i in range(0, math.ceil(len(dict_combined)/100)):
    if (i+1)*100 <= len(dict_combined):
        list_temp = list_conversationid[i*100:(i+1)*100-1]
    elif (i+1)*100 > len(dict_combined):
        list_temp = list_conversationid[i*100:-1]
        
    tweets = t.tweet_lookup(list_temp)
    for search in tweets:
        for tweet in search['data']:
            dict_content[tweet['id']] = tweet

#%%
""" missing tweets? """
list_missconv = []

for conversation_id in list_conversationid:
    if conversation_id not in dict_content.keys():
        list__missconv .append(conversation_id)
        
# get the missing tweets
for i in range(0, math.ceil(len(list__missconv )/100)):
    if (i+1)*100 <= len(list__missconv):
        list_temp = list__missconv [i*100:(i+1)*100-1]
    elif (i+1)*100 > len(list__missconv):
        list_temp = list_temp[i*100:-1]
    
    tweets = t.tweet_lookup(list_temp)
    for search in tweets:
        if 'data' in search:
            for tweet in search['data']:
                dict_content[tweet['id']] = tweet
    
"""
there are a small part of tweets that were protected to not be accessed

try them later by using the OAUTH1.0

"""
#%%
""" conversation screening """
""" filtering out irrelevant conversations """
list_conversationid = list(dict_content.keys()).copy()

list_uncertain = []

for conversation_id in list_conversationid:
    # filter out irrelevant conversations by language or annotation or link
    ## if not in english then delete
    if dict_content[conversation_id]['lang'] != 'en':
        del dict_content[conversation_id]
        continue
    ## if not tag of barack obama, no keyword, and no link, then delete
    if 'obama' not in dict_content[conversation_id]['text'].lower()\
        and ('entities' not in dict_content[conversation_id].keys() or 'urls' not in dict_content[conversation_id]['entities'].keys()):
            del dict_content[conversation_id]
            continue
    ## continue if keyword 'obama arrested' has been found in text
    elif 'obama arrested' in dict_content[conversation_id]['text']:
        continue
    ## continue if url in the tweet is in the list of relevant links
    ### if urls in entities
    if 'entities' in dict_content[conversation_id].keys() and 'urls' in dict_content[conversation_id]['entities'].keys():
        #### if url in entities in list_linkcopy
        for url in dict_content[conversation_id]['entities']['urls']:
            if url['expanded_url'] in list_linkcopy:
                continue
    ## save it if unsure
    else:
        list_uncertain.append(conversation_id)
    




#%%

""" search all of their descendant tweets"""


list_conversationid = list(dict_content.keys())

for conversation_id in list_conversationid:
    # search for its descendants
    if metric in dict_content[conversation_id]['public_metrics'] != 0:
        metric_type = metric[0:-6]
        ## start point of the following search
        since_time = dict_content[conversation_id]['created_at']:
    
    
    
    
    
    
    
    
    
    
    
#%%
list_id = [] # referenced tweet id that are going to be collected in the next step
list_tempid = [] # tweet id where its descendants are failed to be collected  



  
    # search for its descendants
    for metric in dict_combined[conversation_id]['public_metrics']:
        if metric == 'like_count':
            continue
        if dict_combined[conversation_id]['public_metrics'][metric] != 0:
            metric_type = metric[0:-6]
            if tweet_id not in dict_more.keys():
                  dict_more[conversation_id] = {metric_type:{}}
            else: dict_more[conversation_id][metric_type] = {}
            ## start point of the following searching
            since_time = dict_combined[tweet_id]['created_at']
            ## search retweets
            if metric_type == 'retweet':
                ### search retweets of this orginal tweet 
                ### by searching the retweets of the original user (retweets generated by other users)
                query  = 'retweets_of:'+dict_combined[tweet_id]['author_id']
                tweets = t.search_all(query=query, start_time=since_time, max_results=100)
                
                for search in tweets:
                    for tweet in search['data']:
                        if 'referenced_tweets' in tweet.keys():
                            for i in range(len(tweet['referenced_tweets'])):
                                if tweet['referenced_tweets'][i]['type'] == 'retweeted' and \
                                    tweet['referenced_tweets'][i]['id'] == tweet_id:
                                        #### referenced to the original tweet
                                        dict_more[tweet_id][metric_type][tweet['conversation_id']] = tweet

            ## search replies
            elif metric_type == 'reply':
                ### search replies of this original tweet
                ### by searching the replies of the original user (replies generated by other users)
                query = 'to:'+dict_combined[tweet_id]['author_id']
                tweets = t.search_all(query=query, start_time=since_time, max_results=100)
                
                for search in tweets:
                    for tweet in search['data']:
                        if 'referenced_tweets' in tweet.keys():
                            for i in range(len(tweet['referenced_tweets'])):
                                if tweet['referenced_tweets'][i]['type'] == 'replied_to' and \
                                    tweet['referenced_tweets'][i]['id'] == tweet_id:
                                        #### referenced to the original tweet
                                        dict_more[tweet_id][metric_type][tweet['conversation_id']] = tweet_id
            
            ## search quotes
        elif metric_type == 'quote':
            ### search 
      
        
#%%

                                        #### if more descendatns, save the id in the list for next round of searching
                                        if tweet['public_metrics']['retweet_count'] != 0 or \
                                            tweet['public_metrics']['reply_count'] != 0 or \
                                                tweet['public_metrics']['quote_count'] != 0:
                                                    list_id.append(tweet['conversation_id'])   

        
        
        
        
        
        
#%%

""" save json """

path = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\Twitter Fake News\datasets\data collection\obama arrested for espionage'
with open(os.path.join(path, 'tweets_keyword03.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_keyword03, outfile, ensure_ascii=False) 



with open(os.path.join(path, 'tweets_url03.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_url03, outfile, ensure_ascii=False) 
        


with open(os.path.join(path, 'tweets_combined.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_combined, outfile, ensure_ascii=False) 


with open(os.path.join(path, 'tweets_urls.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_urls, outfile, ensure_ascii=False) 

#%%




""" save links """

# list_linkcopy : with relevant links
list_link = []
for link in list_linkcopy:
    list_link.append([link])

with open(os.path.join(path, 'links.csv'), 'w', encoding="utf-8") as outfile:
    write = csv.writer(outfile)
    write.writerows(list_link)

#%%
""" save ids """

# list_conversationid: sonversation ids which have been filtered (qualified conversation ids)
with open(os.path.join(path, 'list_conversationid.text'), 'w', encoding="utf-8") as outfile:
    for id in list_conversationid:
        outfile.write(id)
        outfile.write('\n')


# list_missingconversationid: conversation ids that required OAUTH v1 to collect (been protected by the users)
with open(os.path.join(path, 'list_missconv.text'), 'w', encoding="utf-8") as outfile:
    for id in list_missconv:
        outfile.write(id)
        outfile.write('\n')


# list_uncertain: all the uncertain conversation ids during screening, which should be manually double-checked
with open(os.path.join(path, 'list_uncertain.text'), 'w', encoding="utf-8") as outfile:
    for id in list_uncertain:
        outfile.write(id)
        outfile.write('\n')












#%%
""" backup code """
#%%
""" aggregate links """
list_cls = []
dict_cls = {} 

for link in list_link:
    if 'https://' in link:
        temp_link = link[8:-1]
    elif 'http://' in link:
        temp_link = link[7:-1]
        
    # find domain name of the link
    site_link = temp_link.split('/')[0]
    
    if len(temp_link.split('.')) == 3:
        site = site_link.split('.')[1]
    else:
        site = site_link.split('.')[0]
        
    if site not in dict_cls.keys():
        dict_cls[site] = {'common link':[], 'links': [temp_link]}
    else:
        dict_cls[site]['links'].append(temp_link)

#%%
""" find common links """
for site in dict_cls.keys():
    if len(dict_cls[site]['links']) == 1:
        dict_cls[site]['common link'].append(dict_cls[site]['links'][0])
    else:
        # find common links
        i = 0
        while i < len(dict_cls[site]['links'])-1:
            link_i = dict_cls[site]['links'][i]
            
            for j in (i+1, len(dict_cls[site]['links'])-1):
                link_j = dict_cls[site]['links'][j]
                common_link = os.path.commonprefix([link_i, link_j])
                if len(common_link.split('/')) >= 2 and common_link not in dict_cls[site]['common link']:
                    dict_cls[site]['common link'].append(common_link)
            
            i += 1

#%%
""" tweepy """
#client = tweepy.Client(bearer_token = bearer_token, wait_on_rate_limit = True)
client = tweepy.Client(bearer_token=bearer_token, consumer_key = consumer_key, consumer_secret = consumer_secret,\
                       access_token = access_token , access_token_secret = access_token_secret, wait_on_rate_limit = True)
tweets = client.search_recent_tweets(query=keyword01, tweet_fields=['id','text','author_id','context_annotations','conversation_id',\
                                                                    'created_at','entities','geo','in_reply_to_user_id','public_metrics',\
                                                                        'referenced_tweets'], \
                                     start_time = start_time, end_time = end_time, max_results=100)


#%%
for tweet in tweets:
     print(tweet)
     break








