# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 13:58:39 2021

@author: One
"""

#%% Get the content of a page using the requests library
import requests
mywebpage_url='https://www.sis.uta.fi/~tojape/'
#mywebpage_url='https://www.tuni.fi/en/'
mywebpage_html=requests.get(mywebpage_url)
#%% Parse the HTML content using beautifulsoup
import bs4
mywebpage_parsed=bs4.BeautifulSoup(mywebpage_html.content,'html.parser')
#%% Get the text content of the page
def getpagetext(parsedpage):
# Remove HTML elements that are scripts
    scriptelements=parsedpage.find_all('script')
# Concatenate the text content from all table cells
    for scriptelement in scriptelements:
# Extract this script element from the page.
# This changes the page given to this function!
        scriptelement.extract()
    pagetext=parsedpage.get_text()
    return(pagetext)
mywebpage_text=getpagetext(mywebpage_parsed)
print(mywebpage_text)
# Find HTML elements that are table cells or 'div' cells
# tablecells=parsedpage.find_all(['td','div'])
# Concatenate the text content from all table or div cells
# pagetext=''
# for tablecell in tablecells:
#     pagetext=pagetext+'\n'+tablecell.text.strip()
#%% Find linked pages in Finnish sites, but not PDF or PS files
def getpageurls(webpage_parsed,crawled_urls=[]):
    # Find elements that are hyperlinks
    pagelinkelements=webpage_parsed.find_all('a')
    pageurls=[];
    for pagelink in pagelinkelements:
        pageurl_isok=1
        try: 
            pageurl=pagelink['href']
        except:
            pageurl_isok=0
        if pageurl_isok==1:
        # Check that the url does NOT contain these strings
            if (pageurl.find('.pdf')!=-1)|(pageurl.find('.ps')!=-1):
                pageurl_isok=0
            # Check that the url DOES contain these strings
            if (pageurl.find('http')==-1)|(pageurl.find('.fi')==-1):
                pageurl_isok=0
            #Updadted ------------------------------
            if (pageurl in crawled_urls):
                pageurl_isok=0
        if pageurl_isok==1:
            pageurls.append(pageurl)
    return(pageurls)
mywebpage_urls=getpageurls(mywebpage_parsed)
print(mywebpage_urls)
#%% Basic web crawler
def basicwebcrawler(seedpage_url,maxpages):
    # Store URLs crawled and their text content
    num_pages_crawled=0
    crawled_urls=[]
    crawled_texts=[]
    # Remaining pages to crawl: start from a seed page URL
    pagestocrawl=[seedpage_url]
    # Process remaining pages until a desired number
    # of pages have been found
    while (num_pages_crawled<maxpages)&(len(pagestocrawl)>0):
        # Retrieve the topmost remaining page and parse it
        pagetocrawl_url=pagestocrawl[0]
        print('Getting page-:')
        print(pagetocrawl_url)
        pagetocrawl_html=requests.get(pagetocrawl_url)
        pagetocrawl_parsed=bs4.BeautifulSoup(pagetocrawl_html.content,'html.parser')
        # Get the text and URLs of the page
        pagetocrawl_text=getpagetext(pagetocrawl_parsed)
        #Updated ---------------------
        pagetocrawl_urls=getpageurls(pagetocrawl_parsed,crawled_urls)
        # Store the URL and content of the processed page
        num_pages_crawled=num_pages_crawled+1
        crawled_urls.append(pagetocrawl_url)
        crawled_texts.append(pagetocrawl_text)
        # Remove the processed page from remaining pages,
        # but add the new URLs
        #Updated --------------------------
        pagestocrawl.extend(pagetocrawl_urls[0:10])
        pagestocrawl=pagestocrawl[1:len(pagestocrawl)]
        
    return(crawled_urls,crawled_texts)
mycrawled_urls_and_texts=basicwebcrawler('https://www.sis.uta.fi/~tojape/',10)
mycrawled_urls=mycrawled_urls_and_texts[0]
mycrawled_texts=mycrawled_urls_and_texts[1]
#%% td
mytext=' Look, here are some words!\n Great! '
mytext.strip()
' '.join(mytext.split())
sentenceSplitter=nltk.data.load('tokenizers/punkt/english.pickle')
sentenceSplitter("E.g., J. Smith knows... and I know. But do you?")
nltk.word_tokenize("Hey, what's going on? Who's that?")
mytokenizedtext=nltk.word_tokenize("Hey, what's going on? Who's that?")
mynltktext=nltk.Text(mytokenizedtext)
#%% Tokenize loaded texts and change them to NLTK
format
import nltk
mycrawled_nltktexts=[]
for k in range(len(mycrawled_texts)):
temp_tokenizedtext=nltk.word_tokenize(mycrawled_texts
[k])
temp_nltktext=nltk.Text(temp_tokenizedtext)
mycrawled_nltktexts.append(temp_nltktext)
'Text Processing Stages'.lower()