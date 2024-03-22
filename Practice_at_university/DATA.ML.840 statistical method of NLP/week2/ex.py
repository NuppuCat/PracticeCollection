# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:09:10 2021

@author: One
"""
#%% Basic file crawler
import os
def gettextlist(directory_path):
directory_textfiles=[]
directory_nontextfiles=[]
directory_nonfiles=[]
# Process each item in the directory
directory_contents=os.listdir(directory_path)
for contentitem in directory_contents:
temp_fullpath=os.path.join(directory_path, contentitem)
# Non-files (e.g. subdirectories) are stored separately
if os.path.isfile(temp_fullpath)==0:
directory_nonfiles.append(contentitem)
else:
# Is this a non-text file (not ending in .txt)?
if temp_fullpath.find('.txt')==-1:
directory_nontextfiles.append(contentitem)
else:
# This is a text file
directory_textfiles.append(contentitem)
return(directory_textfiles,directory_nontextfiles,directory_nonfiles)
mydirectory_path='c:/jaakkos_files/work/teaching/tampere_text_analytics'
mydirectory_contentlists=gettextlist(mydirectory_path)

#%% Basic file crawler
def basicfilecrawler(directory_path):
# Store filenames read and their text content
num_files_read=0
crawled_filenames=[]
crawled_texts=[]
directory_contentlists=gettextlist(directory_path)
# In this basic crawled we just process text files
# and do not handle subdirectories
directory_textfiles=directory_contentlists[0]
for contentitem in directory_textfiles:
print('Reading file:')
print(contentitem)
# Open the file and read its contents
temp_fullpath=os.path.join(directory_path, contentitem)
temp_file=open(temp_fullpath,'r',encoding='utf-8',errors='ignore')
temp_text=temp_file.read()
temp_file.close()
# Store the read filename and content
crawled_filenames.append(contentitem)
crawled_texts.append(temp_text)
num_files_read=num_files_read+1
return(crawled_filenames,crawled_texts)
mycrawled_filenames_and_texts=basicfilecrawler('c:/jaakkos_files/work/teaching/
tampere_text_analytics')
mycrawled_filenames=mycrawled_filenames_and_texts[0]
mycrawled_texts=mycrawled_filenames_and_texts[1]
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
tablecells=parsedpage.find_all(['td','div'])
# Concatenate the text content from all table or div
cells
pagetext=''
for tablecell in tablecells:
pagetext=pagetext+'\n'+tablecell.text.strip()
#%% Find linked pages in Finnish sites, but not PDF or PS files
def getpageurls(webpage_parsed):
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
print('Getting page:')
print(pagetocrawl_url)
pagetocrawl_html=requests.get(pagetocrawl_url)
pagetocrawl_parsed=bs4.BeautifulSoup(pagetocrawl_html.content,'html.parser')
# Get the text and URLs of the page
pagetocrawl_text=getpagetext(pagetocrawl_parsed)
pagetocrawl_urls=getpageurls(pagetocrawl_parsed)
# Store the URL and content of the processed page
num_pages_crawled=num_pages_crawled+1
crawled_urls.append(pagetocrawl_url)
crawled_texts.append(pagetocrawl_text)
# Remove the processed page from remaining pages,
# but add the new URLs
pagestocrawl=pagestocrawl[1:len(pagestocrawl)]
pagestocrawl.extend(pagetocrawl_urls)
return(crawled_urls,crawled_texts)
mycrawled_urls_and_texts=basiccrawler('https://www.sis.uta.fi/~tojape/',10)
mycrawled_urls=mycrawled_urls_and_texts[0]
mycrawled_texts=mycrawled_urls_and_texts[1]
mytext=' Look, here are some words!\n Great! '
mytext.strip()
Out: 'Look, here are some words!\n Great!
' '.join(mytext.split())
Out: 'Look, here are some words! Great!'
sentenceSplitter=nltk.data.load('tokenizers/punkt/english.pickle')
sentenceSplitter("E.g., J. Smith knows... and I know. But do you?")
Out: ['E.g., J. Smith knows... and I know.', 'But do you?']
nltk.word_tokenize("Hey, what's going on? Who's that?")
Out: ['Hey', ',', 'what', "'s", 'going', 'on', '?', 'Who', "'s", 'that', '?']
mytokenizedtext=nltk.word_tokenize("Hey, what's going on?
Who's that?")
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
#%% Make all crawled texts lowercase
mycrawled_lowercasetexts=[]
for k in range(len(mycrawled_nltktexts)):
temp_lowercasetext=[]
for l in range(len(mycrawled_nltktexts[k])):
lowercaseword=mycrawled_nltktexts[k][l].lower()
temp_lowercasetext.append(lowercaseword)
temp_lowercasetest=nltk.Text(temp_lowercasetext)
mycrawled_lowercasetexts.append(temp_lowercasetext)
stemmer=nltk.stem.porter.PorterStemmer()
stemmer.stem('modelling')
Out: 'model'
stemmer.stem('incredible')
Out: 'incred
#%% Stem the loaded texts
stemmer=nltk.stem.porter.PorterStemmer()
def stemtext(nltktexttostem):
stemmedtext=[]
for l in range(len(nltktexttostem)):
# Stem the word
wordtostem=nltktexttostem[l]
stemmedword=stemmer.stem(wordtostem)
# Store the stemmed word
stemmedtext.append(stemmedword)
return(stemmedtext)
mycrawled_stemmedtexts=[]
for k in range(len(mycrawled_lowercasetexts)):
temp_stemmedtext=stemtext(mycrawled_lowercasetexts[k])
temp_stemmedtext=nltk.Text(temp_stemmedtext)
mycrawled_stemmedtexts.append(temp_stemmedtext)
# Download wordnet resource if you do not have it already
nltk.download('wordnet')
lemmatizer=nltk.stem.WordNetLemmatizer()
lemmatizer.lemmatize('better','a')
Out[309]: 'good'
lemmatizer.lemmatize('lighter','n')
Out[309]: 'lighter'
lemmatizer.lemmatize('lighter','a')
Out[310]: 'light'
lemmatizer.lemmatize('automated','v')
Out[311]: 'automate'
lemmatizer.lemmatize('automated','a')
Out[312]: 'automated'
# Download tagger resource if you do not have it already
nltk.download('averaged_perceptron_tagger')
text1=nltk.Text(nltk.word_tokenize('it is lighter than before'))
nltk.pos_tag(text1)
Out:
[('it', 'PRP'),
('is', 'VBZ'),
('lighter', 'JJR'), Here 'lighter' is tagged as
('than', 'IN'), a comparative adjective (JJR)
('before', 'IN')]
text2=nltk.Text(nltk.word_tokenize('it is lighter than before'))
nltk.pos_tag(text2)
nltk.pos_tag(nltk.Text(nltk.word_tokenize('it is a lighter that I
bought')))
Out:
[('it', 'PRP'),
('is', 'VBZ'),
('a', 'DT'),
('lighter', 'NN'), Here 'lighter' is tagged as a noun (NN).
('that', 'WDT'), The tagger uses Penn Treebank tags,
('I', 'PRP'), use this to see descriptions:
('bought', 'VBD')] nltk.download('tagsets')
nltk.help.upenn_tagset()
#%% Convert a POS tag for WordNet
def tagtowordnet(postag):
wordnettag=-1
if postag[0]=='N':
wordnettag='n'
elif postag[0]=='V':
wordnettag='v'
elif postag[0]=='J':
wordnettag='a'
elif postag[0]=='R':
wordnettag='r'
return(wordnettag)
#%% POS tag and lemmatize the loaded texts
# Download tagger and wordnet resources if you do not have them already
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
lemmatizer=nltk.stem.WordNetLemmatizer()
def lemmatizetext(nltktexttolemmatize):
# Tag the text with POS tags
taggedtext=nltk.pos_tag(nltktexttolemmatize)
# Lemmatize each word text
lemmatizedtext=[]
for l in range(len(taggedtext)):
# Lemmatize a word using the WordNet converted POS tag
wordtolemmatize=taggedtext[l][0]
wordnettag=tagtowordnet(taggedtext[l][1])
if wordnettag!=-1:
lemmatizedword=lemmatizer.lemmatize(wordtolemmatize,wordnettag)
else:
lemmatizedword=wordtolemmatize
# Store the lemmatized word
lemmatizedtext.append(lemmatizedword)
return(lemmatizedtext)
mycrawled_lemmatizedtexts=[]
for k in range(len(mycrawled_lowercasetexts)):
lemmatizedtext=lemmatizetext(mycrawled_lowercasetexts[k])
lemmatizedtext=nltk.Text(lemmatizedtext)
mycrawled_lemmatizedtexts.append(lemmatizedtext)
#%% Find the vocabulary, in a distributed fashion
import numpy
myvocabularies=[]
myindices_in_vocabularies=[]
# Find the vocabulary of each document
for k in range(len(mycrawled_lemmatizedtexts)):
# Get unique words and where they occur
temptext=mycrawled_lemmatizedtexts[k]
uniqueresults=numpy.unique(temptext,return_inverse=True)
uniquewords=uniqueresults[0]
wordindices=uniqueresults[1]
# Store the vocabulary and indices of document words in it
myvocabularies.append(uniquewords)
myindices_in_vocabularies.append(wordindices)
myvocabularies[0]
Out: array(['!', '$', '&', ..., 'ziyuan', '©', 'âkerlund'],
dtype='<U25')
# Unify the vocabularies.
# First concatenate all vocabularies
tempvocabulary=[]
for k in range(len(mycrawled_lemmatizedtexts)):
tempvocabulary.extend(myvocabularies[k])
# Find the unique elements among all vocabularies
uniqueresults=numpy.unique(tempvocabulary,return_inverse=True)
unifiedvocabulary=uniqueresults[0]
wordindices=uniqueresults[1]
# Translate previous indices to the unified vocabulary.
# Must keep track where each vocabulary started in
# the concatenated one.
vocabularystart=0
myindices_in_unifiedvocabulary=[]
for k in range(len(mycrawled_lemmatizedtexts)):
# In order to shift word indices, we must temporarily
# change their data type to a Numpy array
tempindices=numpy.array(myindices_in_vocabularies[k])
tempindices=tempindices+vocabularystart
tempindices=wordindices[tempindices]
myindices_in_unifiedvocabulary.append(tempindices)
vocabularystart=vocabularystart+len(myvocabularies[k])
unifiedvocabulary[1000:1050]
myindices_in_unifiedvocabulary[1][600:650]
unifiedvocabulary[myindices_in_unifiedvocabulary[1]
[600:650]]