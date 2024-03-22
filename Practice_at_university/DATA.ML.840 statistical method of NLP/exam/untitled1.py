# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 13:55:32 2021

@author: One
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:37:28 2021

@author: One
"""

#%%
import os
# temp_fullpath ='D:/GRAM/MasterProgramme/Tampere/DATA.ML.840 statistical method of NLP/week5&6/4group'

def gettextlist(directory_path):
    directory_textfiles=[]
    directory_nontextfiles=[]
    directory_nonfiles=[]
    a =directory_path
    directory_contents=os.listdir(a)
    for contentitem in directory_contents:
            temp_fullpath=os.path.join(a, contentitem)
            # Non-files (e.g. subdirectories) are stored separately
            contentitemp =contentitem
            if os.path.isfile(temp_fullpath)==0:
                directory_nonfiles.append(contentitemp)
                
            else:
                # Is this a non-text file (not ending in .txt)?
                if temp_fullpath.find('.txt')==-1:
                    directory_nontextfiles.append(contentitemp)
                else:
                # This is a text file
                    directory_textfiles.append(contentitemp)
    return(directory_textfiles,directory_nontextfiles,directory_nonfiles)
mydirectory_path='D:/GRAM/MasterProgramme/Tampere/DATA.ML.840 statistical method of NLP/exam/fairy-tales-pruned-paragraphs'
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
        temp_fullpath=directory_path+'/'+contentitem
        print(temp_fullpath)
        temp_file=open(temp_fullpath,'r',encoding='utf-8',errors='ignore')
        temp_text=temp_file.read()
        temp_text=temp_text.lower()
        temp_file.close()
        # Store the read filename and content
        crawled_filenames.append(contentitem)
        crawled_texts.append(temp_text)
        num_files_read=num_files_read+1
    return(crawled_filenames,crawled_texts)

# mydirectory_path='D:/GRAM/MasterProgramme/Tampere/DATA.ML.840 statistical method of NLP/week5&6/4group'

        
mycrawled_filenames_and_texts=basicfilecrawler(mydirectory_path)
mycrawled_filenames=mycrawled_filenames_and_texts[0]
mycrawled_texts=mycrawled_filenames_and_texts[1]
#%%
import re
import nltk
import numpy
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
stemmer=nltk.stem.porter.PorterStemmer()
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
for k in range(len(mycrawled_texts)):
    
    text = nltk.Text(nltk.word_tokenize(mycrawled_texts[k]))
    lemmatizedtext=lemmatizetext(text)
    lemmatizedtext=nltk.Text(lemmatizedtext)
    mycrawled_lemmatizedtexts.append(lemmatizedtext)
#%%
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
highest_totaloccurrences_indices=numpy.argsort(-1*unifiedvocabulary_totaloccurrencecounts,axis=0)
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



#%%

oldtopruned=[]
tempind=-1
remainingvocabulary = unifiedvocabulary[numpy.squeeze(pruningdecisions!=1)]
for k in range(len(unifiedvocabulary)):
    if pruningdecisions[k]==0:
        tempind=tempind+1
        oldtopruned.append(tempind)
    else:
        oldtopruned.append(-1)
#Create pruned texts
mycrawled_prunedtexts=[]
myindices_in_prunedvocabulary=[]
for k in range(len(mycrawled_lemmatizedtexts)):
    # print(k)
    temp_newindices=[]
    temp_newdoc=[]
    for l in range(len(mycrawled_lemmatizedtexts[k])):
        temp_oldindex=myindices_in_unifiedvocabulary[k][l]
        temp_newindex=oldtopruned[temp_oldindex]
        if temp_newindex!=-1:
            temp_newindices.append(temp_newindex)
            temp_newdoc.append(unifiedvocabulary[temp_oldindex])
    mycrawled_prunedtexts.append(temp_newdoc)
    myindices_in_prunedvocabulary.append(temp_newindices)
#%%
#%%   Create TF-IDF vectors
import scipy
n_docs=len(mycrawled_prunedtexts)
n_vocab=len(remainingvocabulary)
# Matrix of term frequencies
tfmatrix=scipy.sparse.lil_matrix((n_docs,n_vocab))
# Row vector of document frequencies
dfvector=scipy.sparse.lil_matrix((1,n_vocab))

numofterms = []
# Loop over documents
for k in range(n_docs):
# Row vector of which words occurred in this document
    temp_dfvector=scipy.sparse.lil_matrix((1,n_vocab))
    # Loop over words
    for l in range(len(mycrawled_prunedtexts[k])):
        # Add current word to term-frequency count and document-count
        currentword=myindices_in_prunedvocabulary[k][l]
        tfmatrix[k,currentword]=tfmatrix[k,currentword]+1
        temp_dfvector[0,currentword]=1
        
    # Add which words occurred in this document to overall document counts
    numofterms.append(len(numpy.unique(mycrawled_prunedtexts[k],return_inverse=True)[0]))
    dfvector=dfvector+temp_dfvector
# Use the count statistics to compute the tf-idf matrix
tfidfmatrix=scipy.sparse.lil_matrix((n_docs,n_vocab))
#length
divm = (numpy.expand_dims(numpy.array(numofterms),axis=1))
divm[divm==0]=1
ltfmatrix = scipy.sparse.lil_matrix(tfmatrix/divm)
# print(tfmatrix)
# print(ltfmatrix.data)
idfvector= []

    
for k in range(n_vocab):
    idfvector.append(1+numpy.log(((dfvector[0,k]+1)**-1)*n_docs))
# idfvector=1+numpy.log(((dfvector.data+1)**-1)*n_docs)
idfvector = numpy.array(idfvector)
for k in range(n_docs):
    
     # Combine the tf and idf terms 
     #NOTE HERE SHOULD BE MYLTIPLY, NOT *
     tfidfmatrix[k,:]=ltfmatrix[k,:].multiply(idfvector)


import numpy as np
def cosine_similarity(x,y):
    
    num = x.dot(y.T)
    
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    
    return num / denom

td = [0,29,99]
sim = []
for i in td:
    sim = []
    for j in range(n_docs):
        x = tfidfmatrix[i,:]
        x= x.todense()
        
        y = tfidfmatrix[j,:]
        y= y.todense()
        siml = np.array(cosine_similarity(x,y))
        siml = siml[0][0]
        sim.append(siml)
    # print(sim)    
    sim = np.array(sim)
    m = np.sort(sim)
    m=m[-2]
    mi = np.argsort(sim)
    mi= mi[-2]
    print("the doc most similar is doc"+str(mi)+" and the similarity value is "+str(m))
    


















