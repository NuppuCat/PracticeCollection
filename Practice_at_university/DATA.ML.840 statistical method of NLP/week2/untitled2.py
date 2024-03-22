# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 14:48:45 2021

@author: One
"""
#%% Get the content of a page using the requests library
import requests
import nltk
mywebpage_url='https://www.gutenberg.org/browse/scores/top#books-last30'
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
def getpageurls(webpage_parsed,startlabel="",crawled_urls=[]):
    
    # Find elements that are hyperlinks
    pagelinkelements=webpage_parsed.find_all(('a','h2'))
    pageurls=[];
    crawled_texts=[]
    
    for pagelink in pagelinkelements:
        if (startlabel!="") :
            try:
                 if (pagelink['id']!=startlabel):
                     continue
                 else:
                    startlabel=""
        
            except:
                 continue
        pageurl_isok=1
        try: 
            pageurl=pagelink['href']
        except:
            pageurl_isok=0
        if pageurl_isok==1:
        # Check that the url does NOT contain these strings |(pageurl.endswith('.htm')!=-1)
            if ((pageurl.find('.pdf')!=-1)|(pageurl.find('.epub')!=-1)|(pageurl.find('.kindle')!=-1)):
                pageurl_isok=0
            # Check that the url DOES contain these strings 
            if ((pageurl.startswith('/')==-1)&(pageurl.find('http')==-1)):
                pageurl_isok=0
            #Updadted ------------------------------
            if (pageurl in crawled_urls):
                pageurl_isok=0
        if pageurl_isok==1: 
            if (pageurl.startswith('/')==1):
                pageurl = "https://www.gutenberg.org"+pageurl
                
            pageurls.append(pageurl)
            crawled_texts.append(pagelink.contents[0])
    return pageurls,crawled_texts
mywebpage_urls_and_title=getpageurls(mywebpage_parsed,"books-last30")
print(mywebpage_urls_and_title[1])
print(len(mywebpage_urls_and_title[0]))
print(len(mywebpage_urls_and_title[1]))


#%% Basic web crawler
def bookurlcrawler(seedpage_url,k=20,startlabel=""):
    # Store URLs crawled and their text content
    
    # Remaining pages to crawl: start from a seed page URL
    pagestocrawl=[seedpage_url]
    # Process remaining pages until a desired number
    # of pages have been found
    # Retrieve the topmost remaining page and parse it
    pagetocrawl_url=pagestocrawl[0]
    print('Getting page-:')
    print(pagetocrawl_url)
    pagetocrawl_html=requests.get(pagetocrawl_url)
    pagetocrawl_parsed=bs4.BeautifulSoup(pagetocrawl_html.content,'html.parser')
    # Get the text and URLs of the page
    # pagetocrawl_text=getpagetext(pagetocrawl_parsed)
    #Updated ---------------------
    pagetocrawl_urls=getpageurls(pagetocrawl_parsed,startlabel)
    ind = 0
    # for i in range(0,len(pagetocrawl_urls[0])):
    #     if i+1>=len(pagetocrawl_urls[0]):
    #         print("cannot locate")
    #         break
    #     if (pagetocrawl_urls[0][i].endswith("#authors-last30")) & (pagetocrawl_urls[1][i].find("Top 100 EBooks last 30 days")!=-1):
    #         ind = i
    #         break
    # print( pagetocrawl_urls[1])
    aim_urls = pagetocrawl_urls[0][ind:ind+k]
    
    aim_titles = pagetocrawl_urls[1][ind:ind+k]
    return aim_urls,aim_titles

#%% Find linked pages in Finnish sites, but not PDF or PS files
import re

def getbooksurls(webpage_parsed,crawled_urls=[]):
    
    # Find elements that are hyperlinks
    pagelinkelements=webpage_parsed.find_all(('a'))
    bookurls=[]
  
  
    for pagelink in pagelinkelements:
        
        pageurl_isok=1
        try: 
            pageurl=pagelink['href']
        except:
            pageurl_isok=0
        if pageurl_isok==1:
        # Check that the url does NOT contain these strings |(pageurl.endswith('.htm')!=-1)
            if ((pageurl.find('.pdf')!=-1)|(pageurl.find('.epub')!=-1)|(pageurl.find('.kindle')!=-1)):
                pageurl_isok=0
            # Check that the url DOES contain these strings 
            if ((pageurl.find('.txt')==-1)):
                pageurl_isok=0
            #Updadted ------------------------------
            if (pageurl in crawled_urls):
                pageurl_isok=0
        if pageurl_isok==1: 
            if (pageurl.startswith('/')==1):
                pageurl = "https://www.gutenberg.org"+pageurl
            
            bookurls.append(pageurl)
            
    if ( (len(bookurls)!=1)):
        
        # print("----------------warning, may be bugs")
       
        # print(bookurls)
        # print("------------------------------------------------------")
        bookurls=bookurls[0]
    return bookurls
def booktextcrawler(bookstocrawl_urls):
   
    crawled_urls=[]
    crawled_texts=[]
    # Store the URL and content of the processed page
    for url in bookstocrawl_urls:
        # print(url)
        url=requests.get(url)
        url=bs4.BeautifulSoup(url.content,'html.parser')
        txturl = getbooksurls(url)[0]
        # print(txturl)
        paresedtxturl=requests.get(txturl)
        paresedtxturl=bs4.BeautifulSoup(paresedtxturl.content,'html.parser')
        url_text = getpagetext(paresedtxturl)
        crawled_urls.append(txturl)
        # startind = url_text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
        # endind = url_text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
        crawled_texts.append(url_text)
    return   crawled_urls,crawled_texts
    # Remove the processed page from remaining pages,
    # but add the new URLs
    #Updated --------------------------
    
        
    return(crawled_urls,crawled_texts)
startlabel = "books-last30"
mywebpage_urls_and_title=bookurlcrawler(mywebpage_url,20,startlabel)
# print(mywebpage_urls_and_title[0])
# print(mywebpage_urls_and_title[1])
books = booktextcrawler(mywebpage_urls_and_title[0])
# print(books[0][0])
#%%
# print(books[1][0])
for i in range(20):
    print("Bood title:"+mywebpage_urls_and_title[1][i])
    print("Book txt url:"+books[0][i])
#%% 
# lowercaseword=books[1][0].lower()
startstr = "START OF THE PROJECT GUTENBERG EBOOK"
# startind = books[1][0].find(startstr)
endstr = "*** END OF THE PROJECT GUTENBERG EBOOK"

booksnew = []
# print(len(books[1]))
bookstext= books[1]
for book in bookstext:
    # print(book)
    startind = book.find(startstr)
    endtind = book.find(endstr)
    startindp = book[startind:startind+500].find("***")
    booknew = book[(startind+startindp+3):endtind]
    booknew = booknew.lower()
    # print(endtind)
    booksnew.append(booknew)
# print(booksnew[0])
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
#%% 
temp_lowercasetest=nltk.Text(booksnew[0])
# print(temp_lowercasetest)
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# def stemtext(nltktexttostem):
#     stemmedtext=[]
#     for l in range(len(nltktexttostem)):
#         # Stem the word
#         wordtostem=nltktexttostem[l]
#         stemmedword=stemmer.stem(wordtostem)
#         # Store the stemmed word
#         stemmedtext.append(stemmedword)
#     return(stemmedtext)
# mycrawled_stemmedtexts=[]
# for k in range(len(booksnew)):
#     temp_stemmedtext=stemtext(booksnew[k])
#     temp_stemmedtext=nltk.Text(temp_stemmedtext)
#     mycrawled_stemmedtexts.append(temp_stemmedtext)

# print(mycrawled_stemmedtexts)
stemmer=nltk.stem.porter.PorterStemmer()
lemmatizer=nltk.stem.WordNetLemmatizer()
# text1=nltk.Text(nltk.word_tokenize('it is lighter than before'))
# print(nltk.pos_tag(text1))
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
#get lemmatized books, in list
mycrawled_lemmatizedtexts=[]
for k in range(len(booksnew)):
    
    text = nltk.Text(nltk.word_tokenize(booksnew[k]))
    lemmatizedtext=lemmatizetext(text)
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
    #___________________________here return_inverse=True keep the index to rebuild temptxt,smart process
    uniqueresults=numpy.unique(temptext,return_inverse=True)
    uniquewords=uniqueresults[0]
    wordindices=uniqueresults[1]
    # Store the vocabulary and indices of document words in it
    myvocabularies.append(uniquewords)
    myindices_in_vocabularies.append(wordindices)
print(myvocabularies[0])
    #%% 

# Unify the vocabularies.
# First concatenate all vocabularies
#  books all words
tempvocabulary=[]
for k in range(len(mycrawled_lemmatizedtexts)):
    tempvocabulary.extend(myvocabularies[k])
# Find the unique elements among all vocabularies
# But not all index, also keep the index to rebuild the whole books. but the index now is from all books, so need next for loop.
uniqueresults=numpy.unique(tempvocabulary,return_inverse=True)
unifiedvocabulary=uniqueresults[0]
wordindices=uniqueresults[1]
# Translate previous indices to the unified vocabulary.
# Must keep track where each vocabulary started in
# the concatenated one.
vocabularystart=0
# keep words index for each book in a list
myindices_in_unifiedvocabulary=[]
for k in range(len(mycrawled_lemmatizedtexts)):
    # In order to shift word indices, we must temporarily
    # change their data type to a Numpy array
    tempindices=numpy.array(myindices_in_vocabularies[k])
    tempindices=tempindices+vocabularystart
    tempindices=wordindices[tempindices]
    myindices_in_unifiedvocabulary.append(tempindices)
    vocabularystart=vocabularystart+len(myvocabularies[k])
print(unifiedvocabulary[4000:4050])
print(myindices_in_unifiedvocabulary[1][4000:4050])
print(unifiedvocabulary[myindices_in_unifiedvocabulary[1][4000:4050]])
#%%
# each word count for all books
unifiedvocabulary_totaloccurrencecounts=numpy.zeros((len(unifiedvocabulary),1))
# each word appear in how many books
unifiedvocabulary_documentcounts=numpy.zeros((len(unifiedvocabulary),1))
# mean of each word appear in every book
unifiedvocabulary_meancounts=numpy.zeros((len(unifiedvocabulary),1))
# variance of each word in every book
unifiedvocabulary_countvariances=numpy.zeros((len(unifiedvocabulary),1))

# First pass: count occurrences
for k in range(len(mycrawled_lemmatizedtexts)):
    print(k)
    occurrencecounts=numpy.zeros((len(unifiedvocabulary),1))
    #for each word in every book, count its appear
    for l in range(len(myindices_in_unifiedvocabulary[k])):
        occurrencecounts[myindices_in_unifiedvocabulary[k][l]]= occurrencecounts[myindices_in_unifiedvocabulary[k][l]]+1
    unifiedvocabulary_totaloccurrencecounts= unifiedvocabulary_totaloccurrencecounts+occurrencecounts
    unifiedvocabulary_documentcounts= unifiedvocabulary_documentcounts+(occurrencecounts>0)
# Mean occurrence counts over documents
unifiedvocabulary_meancounts= unifiedvocabulary_totaloccurrencecounts/len(mycrawled_lemmatizedtexts)
# Second pass to count variances
for k in range(len(mycrawled_lemmatizedtexts)):
    print(k)
    occurrencecounts=numpy.zeros((len(unifiedvocabulary),1))
    for l in range(len(myindices_in_unifiedvocabulary[k])):
        occurrencecounts[myindices_in_unifiedvocabulary[k][l]]= occurrencecounts[myindices_in_unifiedvocabulary[k][l]]+1
    unifiedvocabulary_countvariances=unifiedvocabulary_countvariances+  (occurrencecounts-unifiedvocabulary_meancounts)**2
unifiedvocabulary_countvariances= unifiedvocabulary_countvariances/(len(mycrawled_lemmatizedtexts)-1)

#%% Inspect frequent words
# Sort words by largest total (or mean) occurrence count
# show the words and its counts
highest_totaloccurrences_indices=numpy.argsort(-1*unifiedvocabulary_totaloccurrencecounts,axis=0)
print(numpy.squeeze(unifiedvocabulary[highest_totaloccurrences_indices[0:100]]))
print(numpy.squeeze(unifiedvocabulary_totaloccurrencecounts[highest_totaloccurrences_indices[0:100]]))
#%%  Pruning
# download the stopword list if you do not have it already
nltk.download('stopwords')
nltkstopwords=nltk.corpus.stopwords.words('english')
pruningdecisions=numpy.zeros((len(unifiedvocabulary),1))
for k in range(len(unifiedvocabulary)):
    # Rule 1: check the nltk stop word list
    if (unifiedvocabulary[k] in nltkstopwords):
        pruningdecisions[k]=1
    # Rule 2: if the word is in the top 1% of frequent words
    if (k in highest_totaloccurrences_indices[\
        0:int(numpy.floor(len(unifiedvocabulary)*0.01))]):
        pruningdecisions[k]=1
    # # Rule 3: if the word is in the bottom 65% of frequent words
    # if (k in highest_totaloccurrences_indices[(int(numpy.floor(\
    #     len(unifiedvocabulary)*0.35))):len(unifiedvocabulary)]):
    #     pruningdecisions[k]=1
    # Rule 4: if the word is too short
    if len(unifiedvocabulary[k])<2:
        pruningdecisions[k]=1
    # Rule 5: if the word is too long
    if len(unifiedvocabulary[k])>20:
        pruningdecisions[k]=1
    # Rule 6: if the word has unwanted characters
    # (here for simplicity only a-z allowed)
    # if unifiedvocabulary[k].isalpha()==False:
    #     pruningdecisions[k]=1
    # Rule 7, if the word courred less than 4 times.
    if (unifiedvocabulary_totaloccurrencecounts[k]<4):
        pruningdecisions[k]=1
pruned_highest_totaloccurrences_indices = numpy.argsort(numpy.multiply(-1*unifiedvocabulary_totaloccurrencecounts,(pruningdecisions!=1)),axis=0)
# squeeze:Remove axes of length one from a.
print(numpy.squeeze(unifiedvocabulary[pruned_highest_totaloccurrences_indices[0:100]]))
print(numpy.squeeze(unifiedvocabulary_totaloccurrencecounts[pruned_highest_totaloccurrences_indices[0:100]]))
astemp= -1*unifiedvocabulary_totaloccurrencecounts*pruningdecisions
#%% Make a frequency plot of the words
# Import the plotting library
import matplotlib.pyplot
# # Tell the library we want each plot in its own window
# matplotlib auto
# Create a figure and an axis
matplotlib.pyplot.figure()
myfigure, myaxes = matplotlib.pyplot.subplots();
# Plot the sorted occurrence counts of the words against their ranks
horizontalpositions=range(len(unifiedvocabulary))
verticalpositions1=numpy.squeeze(unifiedvocabulary_totaloccurrencecounts[\
highest_totaloccurrences_indices])/numpy.sum(unifiedvocabulary_totaloccurrencecounts)
myaxes.plot(horizontalpositions,verticalpositions1);
# Zipf's law

matplotlib.pyplot.figure()
myfigure, myaxes = matplotlib.pyplot.subplots();
# Plot the sorted occurrence counts of the words against their ranks
horizontalpositions=range(len(unifiedvocabulary))[:10]
verticalpositions=(numpy.squeeze(unifiedvocabulary_totaloccurrencecounts[\
highest_totaloccurrences_indices])/numpy.sum(unifiedvocabulary_totaloccurrencecounts))[:10]
myaxes.plot(horizontalpositions,verticalpositions,label='original');




horizontalpositions=range(len(unifiedvocabulary))[:10]
# for whole vocabulary list, Zipf only depends on the size of the vocabulary
def getpw(alpha,unifiedvocabulary,highest_totaloccurrences_indices):
    p = []
    n = numpy.array(range(len(unifiedvocabulary)))+1
    bot = numpy.sum(1/(n**alpha))
    
    for  k in range(len(unifiedvocabulary)):
        # rank = numpy.where(unifiedvocabulary[highest_totaloccurrences_indices]==unifiedvocabulary[k])[0][0]
        rank = k +1
        top = 1/(rank**alpha)
        p.append(top/bot)
    return p
# p = getpw(1.2,unifiedvocabulary,highest_totaloccurrences_indices)
# verticalpositions=numpy.squeeze(p)[:200]
# myaxes.plot(horizontalpositions,verticalpositions);
costs = []
for a in range(20):
    a = a*0.1
    p = getpw(a,unifiedvocabulary,highest_totaloccurrences_indices)
    cost=numpy.sum((numpy.squeeze(p)- verticalpositions1)**2)
    costs.append(cost)
    verticalpositions=numpy.squeeze(p)[:10]
    myaxes.plot(horizontalpositions,verticalpositions,label=['a=%f' % a]);
    matplotlib.pyplot.legend()
matplotlib.pyplot.show()
besta = 0.1*numpy.argmin(costs)
print("best a is %f" % besta)
#%% EX3 1 -------------------------------------------------------------------------
# print(booksnew[0])
mycrawled_lemmatizedtexts[0].dispersion_plot(['pride', 'prejudice',
'elizabeth', 'darcy', 'charlotte', 'love', 'hate', 'marriage', 'husband', 'wife', 'father', 'mother',
'daughter', 'family', 'dance', 'happy'])
#%% EX3 2
# print(booksnew[1])
# print("-----------------------------")
words_toconcordance = ['science', 'horror', 'monster', 'fear']
for word in words_toconcordance:
    
    mycrawled_lemmatizedtexts[1].concordance(word)
    
#%% EX3 3
temp_fullpath ='D://GRAM//MasterProgramme//Tampere//DATA.ML.840 statistical method of NLP//week3//215-0.txt'

temp_file=open(temp_fullpath,'r',encoding='utf-8',errors='ignore')
wild=temp_file.read()
temp_file.close()
# print(temp_text)
# downloaded str will change
startstr1 = "START OF THIS PROJECT GUTENBERG EBOOK"
endstr1 = "*** END OF THIS PROJECT GUTENBERG EBOOK"
startind1 = wild.find(startstr1)
endtind1 = wild.find(endstr1)
# print(startstr)
startindp1 = book[startind:startind+500].find("***")
wild = wild[(startind1+startindp1+3):endtind1]
wild = wild.lower()
# print(wild)
wildtext = nltk.Text(nltk.word_tokenize(wild))
wildlemmatizedtext=lemmatizetext(wildtext)
wildlemmatizedtext=nltk.Text(wildlemmatizedtext)
wilduniqueresults=numpy.unique(wildlemmatizedtext,return_inverse=True)
wilduniquewords=wilduniqueresults[0]
wildwordindices=wilduniqueresults[1]
# each word count for book
wildunifiedvocabulary_totaloccurrencecounts=numpy.zeros((len(wilduniquewords),1))
wildoccurrencecounts=numpy.zeros((len(wilduniquewords),1))
for l in range(len(wildwordindices)):
        wildoccurrencecounts[wildwordindices[l]]= wildoccurrencecounts[wildwordindices[l]]+1
wildunifiedvocabulary_totaloccurrencecounts= wildunifiedvocabulary_totaloccurrencecounts+wildoccurrencecounts
wildhighest_totaloccurrences_indices=numpy.argsort(-1*wildunifiedvocabulary_totaloccurrencecounts,axis=0)
print(numpy.squeeze(wilduniquewords[wildhighest_totaloccurrences_indices[0:100]]))
print(numpy.squeeze(wildunifiedvocabulary_totaloccurrencecounts[wildhighest_totaloccurrences_indices[0:100]]))

wildpruningdecisions=numpy.zeros((len(wilduniquewords),1))
for k in range(len(wilduniquewords)):
    # Rule 1: check the nltk stop word list
    if (wilduniquewords[k] in nltkstopwords):
        wildpruningdecisions[k]=1
    # Rule 2: if the word is in the top 1% of frequent words
    if (k in wildhighest_totaloccurrences_indices[\
        0:int(numpy.floor(len(wilduniquewords)*0.005))]):
        wildpruningdecisions[k]=1
    # # Rule 3: if the word is in the bottom 65% of frequent words
    # if (k in highest_totaloccurrences_indices[(int(numpy.floor(\
    #     len(unifiedvocabulary)*0.35))):len(unifiedvocabulary)]):
    #     pruningdecisions[k]=1
    # Rule 4: if the word is too short
    if len(wilduniquewords[k])<2:
        wildpruningdecisions[k]=1
    # Rule 5: if the word is too long
    if len(wilduniquewords[k])>20:
        wildpruningdecisions[k]=1
    # Rule 6: if the word has unwanted characters
    # (here for simplicity only a-z allowed)
    # if unifiedvocabulary[k].isalpha()==False:
    #     pruningdecisions[k]=1
    # Rule 7, if the word courred less than 4 times.
    if (wildunifiedvocabulary_totaloccurrencecounts[k]<4):
        wildpruningdecisions[k]=1
wildpruned_highest_totaloccurrences_indices = numpy.argsort(numpy.multiply(-1*wildunifiedvocabulary_totaloccurrencecounts,(wildpruningdecisions!=1)),axis=0)
# squeeze:Remove axes of length one from a.
print(numpy.squeeze(wilduniquewords[wildpruned_highest_totaloccurrences_indices[0:100]]))
print(numpy.squeeze(wildunifiedvocabulary_totaloccurrencecounts[wildpruned_highest_totaloccurrences_indices[0:100]]))
#%% Get indices of documents to remaining words
# unifiedvocabulary = wilduniquewords
# pruningdecisions = wildpruningdecisions
oldtopruned=[]
tempind=-1
for k in range(len(wilduniquewords)):
    if wildpruningdecisions[k]==0:
        tempind=tempind+1
        oldtopruned.append(tempind)
    else:
        oldtopruned.append(-1)
#%% Create pruned texts
mycrawled_prunedtexts=[]
myindices_in_prunedvocabulary=[]

temp_newindices=[]
temp_newdoc=[]
for l in range(len(wilduniquewords)):
    temp_oldindex=wildwordindices[l]
    temp_newindex=oldtopruned[temp_oldindex]
    if temp_newindex!=-1:
        temp_newindices.append(temp_newindex)
        temp_newdoc.append(wilduniquewords[temp_oldindex])
mycrawled_prunedtexts.append(temp_newdoc)
myindices_in_prunedvocabulary.append(temp_newindices)
print(mycrawled_prunedtexts)
#%% Compute statistics of word distances
# Compute counts and subs of distances and squared distances
import scipy
remainingvocabulary = mycrawled_prunedtexts[0]
distanceoccurrences=scipy.sparse.lil_matrix((len(remainingvocabulary),len(remainingvocabulary)))
sumdistances=scipy.sparse.lil_matrix((len(remainingvocabulary),len(remainingvocabulary)))
sumabsdistances=scipy.sparse.lil_matrix((len(remainingvocabulary),len(remainingvocabulary)))
sumdistancesquares=scipy.sparse.lil_matrix((len(remainingvocabulary),len(remainingvocabulary)))

for l in range(1):
    latestoccurrencepositions=scipy.sparse.lil_matrix(\
    (len(remainingvocabulary),len(remainingvocabulary)))
    # Loop through all word positions m of document l
    for m in range(len(mycrawled_prunedtexts[l])):
        # Get the vocabulary index of the current word in position m
        currentword=myindices_in_prunedvocabulary[l][m]
        # Loop through previous words, counting back up to 10 words from current word
        windowsize=min(m,10)
        for n in range(windowsize):
            # Get the vocabulary index of the previous word in position m-n-1
            previousword=myindices_in_prunedvocabulary[l][m-n-1]
            # Is this the fist time we have encountered this word while
            # counting back from the word at m? Then it is the closest pair.
            if latestoccurrencepositions[currentword,previousword]<m:
                # Store the occurrence of this word pair with the word at m as the 1st word
                distanceoccurrences[currentword,previousword]=\
                distanceoccurrences[currentword,previousword]+1
                sumdistances[currentword,previousword]=sumdistances[\
                currentword,previousword]+((m-n-1)-m)
                sumabsdistances[currentword,previousword]=\
                sumabsdistances[currentword,previousword]+abs((m-n-1)-m)
                sumdistancesquares[currentword,previousword]=\
                sumdistancesquares[currentword,previousword]+((m-n-1)-m)**2
                # Store the occurrence of this word pair with the word at n as the 1st word
                distanceoccurrences[previousword,currentword]=\
                distanceoccurrences[previousword,currentword]+1
                sumdistances[previousword,currentword]=sumdistances[\
                previousword,currentword]+(m-(m-n-1))
                sumabsdistances[previousword,currentword]=\
                sumabsdistances[currentword,previousword]+abs(m-(m-n-1))
                sumdistancesquares[previousword,currentword]=\
                sumdistancesquares[previousword,currentword]+(m-(m-n-1))**2
                # Mark that we found this pair while counting down from m,
                # so we do not count more distant occurrences of the pair
                latestoccurrencepositions[currentword,previousword]=m
                latestoccurrencepositions[previousword,currentword]=m

# Compute distribution statistics based on the counts
n_vocab=len(remainingvocabulary)
distancemeans=scipy.sparse.lil_matrix((n_vocab,n_vocab))
absdistancemeans=scipy.sparse.lil_matrix((n_vocab,n_vocab))
distancevariances=scipy.sparse.lil_matrix((n_vocab,n_vocab))
absdistancevariances=scipy.sparse.lil_matrix((n_vocab,n_vocab))
for m in range(n_vocab):
    # print(m)
    # Find the column indices that have at least two occurrences
    tempindices=numpy.nonzero(distanceoccurrences[m,:]>1)[1]
    # The occurrence vector needs to be a non-sparse data type
    tempoccurrences=distanceoccurrences[m,tempindices].todense()
    # Estimate mean of m-n distance
    distancemeans[m,tempindices]=numpy.squeeze(\
    numpy.array(sumdistances[m,tempindices]/tempoccurrences))
    absdistancemeans[m,tempindices]=numpy.squeeze(\
    numpy.array(sumabsdistances[m,tempindices]/tempoccurrences))
    # Estimate variance of m-n distance
    meanterm=distancemeans[m,tempindices].todense()
    meanterm=numpy.multiply(meanterm,meanterm)
    meanterm=numpy.multiply(tempoccurrences/(tempoccurrences-1),meanterm)
    distancevariances[m,tempindices]=numpy.squeeze(\
    numpy.array(sumdistancesquares[m,tempindices]/(tempoccurrences-1) \
    - meanterm))
    meanterm=absdistancemeans[m,tempindices].todense()
    meanterm=numpy.multiply(meanterm,meanterm)
    meanterm=numpy.multiply(tempoccurrences/(tempoccurrences-1),meanterm)
    absdistancevariances[m,tempindices]=numpy.squeeze(\
    numpy.array(sumdistancesquares[m,tempindices]/(tempoccurrences-1) \
    - meanterm))

# Compute overall distance distribution
overalldistancecount=numpy.sum(distanceoccurrences)
overalldistancesum=numpy.sum(sumdistances)
overallabsdistancesum=numpy.sum(sumabsdistances)
overalldistancesquaresum=numpy.sum(sumdistancesquares)
overalldistancemean=overalldistancesum/overalldistancecount
overallabsdistancemean=overallabsdistancesum/overalldistancecount
overalldistancevariance=overalldistancesquaresum/(overalldistancecount-1)\
-overalldistancecount/(overalldistancecount-1)*overalldistancemean
overallabsdistancevariance=overalldistancesquaresum/(overalldistancecount-1)\
-overalldistancecount/(overalldistancecount-1)*overallabsdistancemean


#%% Compute t-test pvalues comparing abs distance distributions
absdistancepvalues=scipy.sparse.lil_matrix((n_vocab,n_vocab))
for m in range(n_vocab):
    # Find pairs of word m
    tempindices=numpy.nonzero(distanceoccurrences[m,:]>1)[1]
    # For computation we need to transform these to non-sparse vectors
    meanterm=absdistancemeans[m,tempindices].todense()
    varianceterm=absdistancevariances[m,tempindices].todense()
    occurrenceterm=distanceoccurrences[m,tempindices].todense()
    # Compute the t-test statistic for each pair
    tempstatistic=(meanterm-overallabsdistancemean)/ \
    numpy.sqrt(varianceterm/occurrenceterm+ \
    overallabsdistancevariance/overalldistancecount)
    # Compute the t-test degrees of freedom for each pair
    tempdf=(numpy.power(varianceterm/occurrenceterm+\
    overallabsdistancevariance/overalldistancecount,2))/ \
    ( (numpy.power(varianceterm/occurrenceterm,2))/(occurrenceterm-1)+ \
    ((overallabsdistancevariance/overalldistancecount)**2)/ \
    (overalldistancecount-1) )
    # Compute the t-test p-value for each pair
    temppvalue=scipy.stats.t.cdf(tempstatistic,tempdf)
    # Store the t-test p-value for each pair
    absdistancepvalues[m,tempindices]=numpy.squeeze(numpy.array(temppvalue))
#%% Sort word pairs of a particular word by minimum mean absolute distance
def findwordindex(wordstring):
    for k in range(len(remainingvocabulary)):
        if remainingvocabulary[k]==wordstring:
            return(k)
    return(-1)
def printtopcollocations(wordstring):
    # Find the chosen word and words that occurred with it at least 2 times
    mywordindex=findwordindex(wordstring)
    if mywordindex==-1:
        print('Word not found: '+wordstring)
        return
    # Require at least 10 pair occurrences
    minpairoccurrences=10
    tempindices=numpy.nonzero(distanceoccurrences[mywordindex,:]>minpairoccurrences)[1]
    # Sort the pairs by lowest pvalue
    lowest_meandistances_indices=numpy.argsort(numpy.squeeze(\
    numpy.array(absdistancepvalues[mywordindex,tempindices].todense())),axis=0)
    # Print the top-50 lowest-distance pairs
    print('\nLowest p-values\n')
    for k in range(min(50,len(lowest_meandistances_indices))):
        otherwordindex=tempindices[lowest_meandistances_indices[k]]
        # Print the words, their absolute distances (mean+std) and distances (mean+std)
        print('{!s}--{!s}: {:d} occurrences, absdist: {:.1f} +- {:.1f}, offset: {:.1f} +- {:.1f}, pvalue: {:f}'.format(\
        remainingvocabulary[mywordindex],\
        remainingvocabulary[otherwordindex],\
        int(distanceoccurrences[mywordindex,otherwordindex]),\
        absdistancemeans[mywordindex,otherwordindex],\
        numpy.sqrt(absdistancevariances[mywordindex,otherwordindex]),\
        distancemeans[mywordindex,otherwordindex],\
        numpy.sqrt(distancevariances[mywordindex,otherwordindex]),\
        absdistancepvalues[mywordindex,otherwordindex]))
#%% 

arows = absdistancemeans.rows
abdma = absdistancemeans.data
wilddic = dict()

for i in range(len(arows)):
    if (len(arows[i])!=0):
        text = nltk.Text(nltk.word_tokenize(remainingvocabulary[i]))
        taggedtext=nltk.pos_tag(text)
        wordnettag=tagtowordnet(taggedtext[0][1])
        
        if wordnettag=='n' or wordnettag=='a':
            
            for j in range(len(arows[i])):
                text2 = nltk.Text(nltk.word_tokenize(remainingvocabulary[arows[i][j]]))
                taggedtext2=nltk.pos_tag(text2)
                wordnettag2=tagtowordnet(taggedtext2[0][1])
                if wordnettag2=='n':
                    
                    wilddic[abdma[i][j]] = [remainingvocabulary[i],remainingvocabulary[arows[i][j]],absdistancepvalues[i,arows[i][j]]] 
import collections

od = collections.OrderedDict(sorted(wilddic.items()))                
# print(abdma)
# print(absdistancemeans[abdma])
c = 0
top20 =[]
for item in od:
    if c>20:
        break
    # print(remainingvocabulary((od[item][0])))
    # top20.append([(od[item][0])]+' '+remainingvocabulary[(od[item][1])])
    top20.append(od[item])
    c = c+1
print('the lowest 20 collocations are')
print(top20)

#%% 
c = 0
top20dogs = []
for item in od:
    if c>20:
        break
    # print(remainingvocabulary((od[item][0])))
    # top20.append([(od[item][0])]+' '+remainingvocabulary[(od[item][1])])
    label = False
    for it in od[item]:
        # print(it)
        if it=='dog':
            label= True
    if label:    
        top20dogs.append(od[item])
        c = c+1

print(top20dogs)
#%% Ex3 3.4
Frankenstein = booksnew[1]
import re
pattern = "for [\w\s]* years"
pattern=re.compile(pattern)
allmatches=re.finditer(pattern,Frankenstein)
for tempmatch in allmatches:
    print(tempmatch.group(),tempmatch.span())


















