# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:23:19 2020

@author: yjian
"""


"""
This is a file for setting up web crawler for 
collecting links of fake news from fact-check websites

"""

#%%
from requests_html import HTMLSession
import requests_html
import requests
requests.packages.urllib3.disable_warnings() # ignore connection warning
import os
import json

#%%
dict_links = {}

#%%
""" --- web crawler --- """

session = HTMLSession()
#url = 'https://www.politifact.com/personalities/blog-posting/'
url = 'https://www.politifact.com/factchecks/list/?speaker=blog-posting'
h = session.get(url)


#%%
""" abtain all links from current url """
h.html.absolute_links 


#%%
""" save all links related to fake news published in bloggers """


list_articles = h.html.find('article[class^="m-statement"]')

#%%
for article in list_articles:
   if len(article.find('a[href^="/factchecks/2020/"][href*="blog-posting"]')) != 0:
       
       element = article.find('a[href^="/factchecks/2020/"]')[0]
       img = article.find('img[src*="meter"][class*="original"]')[0]
           
       index = list(element.links)[0].strip('/factchecks').strip(list(element.links)[0].split('/')[-2])
       
       # check if index need to be updated
       ## check if in the dict
       if index in dict_links.keys():
           ### check title
           if dict_links[index]['title'] == element.text:
               continue # current news has been saved already
           else:# check how many similar indices have been saved
               index = index + str(len([i for i in list(dict_links.keys()) if str(index) in i])+1)
           
       # get article text
       for i in range(0,10):
           try:
               temp = session.get(list(element.absolute_links)[0])
               
               print('status:', temp.status_code)
               if len(temp.text) > 0:
                   break
           except (requests.exceptions.ConnectionError,NameError):
               continue
       
       #rindex = list(temp.html.find('article[class^="m-textblock"]'))[0].find('p>a[href*="archive"]')[0].absolute_links[0].rindex('https')
       
       dict_links[index] = {'absolute_links': list(element.absolute_links)[0],\
                            'title': element.text,\
                            'level': img.attrs['alt'],\
                            'statement': temp.html.find('h2[class*="c-title c-title--subline"]')[0].text,\
                            'article': temp.html.find('article[class*="m-textblock"]')[0].text,\
                            #'source_archive': list(temp.html.find('article[class^="m-textblock"]')[0].find('p>a[href*="archive"]')[0].absolute_links)[0],\
                            #'source': list(temp.html.find('article[class*="m-superbox"]')[0].find('p:first_child')[0].absolute_links)[0],\
                            'source': list(temp.html.find('article[class*="m-superbox"]')[0].find('p>a')[0].absolute_links)[0],\
                            'inserted_links': list(temp.html.find('article[class*="m-textblock"]')[0].absolute_links)}
           
    
#%%
""" save dict_links """

path = r'C:\Users\yjian\OneDrive\Documents\research files\dissertation\datasets\data collection\politifact'
with open(os.path.join(path, 'fake news_blog-posting.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_links, outfile, ensure_ascii=False) 






#%%
temp = session.get(list(element.absolute_links)[0])

#%%
""" other code """


#h.html.text     
#h.html.html     
#h.html.links   

import requests 
from bs4 import BeautifulSoup

url = 'https://www.politifact.com/personalities/blog-posting/'

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64)\
 AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
 
strhtml = requests.get(url, headers = headers) #get web data
print(strhtml.text)     

s = BeautifulSoup(strhtml.text, "html.parser")
s.findAll('a', {'class':'s-access-detail-page'})