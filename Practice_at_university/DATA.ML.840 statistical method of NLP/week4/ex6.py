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
    dirnames = ['rec.autos', 'rec.motorcycles','rec.sport.baseball',  'rec.sport.hockey']
    for dirname in dirnames:
        # Process each item in the directory
        a = directory_path+'/'+dirname
        print(a)
        directory_contents=os.listdir(a)
        for contentitem in directory_contents:
            temp_fullpath=os.path.join(a, contentitem)
            # Non-files (e.g. subdirectories) are stored separately
            contentitemp = '/'+dirname+'/'+contentitem
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
mydirectory_path='D:/GRAM/MasterProgramme/Tampere/DATA.ML.840 statistical method of NLP/week5&6/4group'
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
    directory_textfiles=directory_contentlists[1]
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
#%%LSA
dimensiontotals=numpy.squeeze( \
numpy.array(numpy.sum(tfidfmatrix,axis=0)))
highesttotals=numpy.argsort(-1*dimensiontotals)
Xsmall=tfidfmatrix[:,highesttotals[0:500]]
#%%
# Xsmall=Xsmall[9000:10000,:].todense()
# Compute 10 factors from LSA
n_low_dimensions=10
Uleft,D,UrightT=scipy.sparse.linalg.svds(\
Xsmall,k=n_low_dimensions)
    # Examine the singular values
print(D)

# # Examine a factor (here the one with largest singular value)
# print(UrightT[9,:])

# # 20 words with largest absolute weights in the factor
for i in range(n_low_dimensions):
    print('this is factor number:'+str(i))
    topweights_indices=numpy.argsort(-1*numpy.abs(UrightT[i,:]))
    print(remainingvocabulary[highesttotals[topweights_indices[0:10]]])

#%%
n_low_dimensions=15
Uleft,D,UrightT=scipy.sparse.linalg.svds(\
Xsmall,k=n_low_dimensions)

print(D)

for i in range(10,n_low_dimensions):
    print('this is factor number:'+str(i))
    topweights_indices=numpy.argsort(-1*numpy.abs(UrightT[i,:]))
    print(remainingvocabulary[highesttotals[topweights_indices[0:10]]])

#%%6.2

import numpy, numpy.matlib, scipy, scipy.stats
def plsa(document_to_word_matrix, n_topics, n_iterations):
    n_docs=numpy.shape(document_to_word_matrix)[0] # Number of documents and vocabulary words
    n_vocab=numpy.shape(document_to_word_matrix)[1]
    theta = scipy.stats.uniform.rvs(size=(n_vocab,n_topics)) # Prob of words per topic: random init
    theta = theta/numpy.matlib.repmat(numpy.sum(theta,axis=0),n_vocab,1)
    psi = scipy.stats.uniform.rvs(size=(n_topics,n_docs)) # Probs topics per document: random init
    psi = psi/numpy.matlib.repmat(numpy.sum(psi,axis=0),n_topics,1)
    n_words_in_docs = numpy.squeeze(numpy.array(numpy.sum(document_to_word_matrix,axis=1))) # Numbers of words in documents: computed once

    n_totalwords = numpy.sum(n_words_in_docs) # Total number of words: computed once
    pi = n_words_in_docs/n_totalwords # Document probs: computed once
    # document_to_word_matrix_t =numpy.ones((n_docs,n_vocab))
    # myepsilon=1 # Add a positive number to avoid divisions by zero
    # for i in range(n_docs):
    #     for j in range(n_vocab):
    #         document_to_word_matrix_t[i,j]=document_to_word_matrix[i,j]+document_to_word_matrix_t[i,j]
    for myiter in range(n_iterations): # Perform Expectation-Maximization iterations
        print(myiter)
        # ===Perform E-step====
        doc_word_to_topics = [] # Compute theta_{v|t}psi_{t|d}/sum_t' theta_{v|t'}psi_{t'|d}
        doc_word_to_topic_sum = numpy.zeros((n_docs,n_vocab))
        for t in range(n_topics):
            mat=  numpy.matlib.repmat(theta[:,t],n_docs,1)
            mat2 = numpy.matlib.repmat(psi[t,:],n_vocab,1)
            doc_word_to_topict = mat * mat2.T
            myepsilon=1e-14
            doc_word_to_topict += myepsilon
            doc_word_to_topics.append(doc_word_to_topict)
            doc_word_to_topic_sum += doc_word_to_topict
        print('step1 f')
        for t in range(n_topics):
            doc_word_to_topics[t] /= doc_word_to_topic_sum
        # =======Perform M-step=======
        # Add a small number to word counts to avoid divisions by zero
        print('step2 f')
        for t in range(n_topics): # Compute document-to-topic probabilities.
            psi[t,:] = numpy.squeeze(numpy.array(numpy.sum( \
            numpy.multiply(document_to_word_matrix+myepsilon,doc_word_to_topics[t]),axis=1)))
        psi /= numpy.matlib.repmat(numpy.sum(psi,axis=0),n_topics,1)
        for t in range(n_topics): # Compute topic-to-word probabilities
            theta[:,t]= numpy.squeeze(numpy.array(numpy.sum( \
            numpy.multiply(document_to_word_matrix,doc_word_to_topics[t]),axis=0).T))
        theta /= numpy.matlib.repmat(numpy.sum(theta,axis=0),n_vocab,1)
    return(pi,psi,theta)


#%% Try PLSA on a very small data set
# Let's take the same 500 features as in LSI,
# and the same 1000 baseball documents but use
# the term-frequency (TF) values for the features



dimensiontotals=numpy.squeeze( \
numpy.array(numpy.sum(tfmatrix,axis=0)))
highesttotals=numpy.argsort(-1*dimensiontotals)
Xsmall=tfmatrix[:,highesttotals[0:500]]
#!!!!!!!!!!!!!!!!!! todense is important to imporve the speed
Xsmall=Xsmall[:,:].todense()
#%%

# Run PLSA
n_topics=10
n_iterations=10
pi,psi,theta=plsa(Xsmall, n_topics, n_iterations)
#%%
# Examine the factor probabilities p(t) = sum_d p(t|d) p(d)
print(numpy.sum(psi*numpy.matlib.repmat(pi,n_topics,1),axis=1))
for i in range(n_topics):
    topweights_indices=numpy.argsort(-1*numpy.abs(theta[:,i]))
    print(remainingvocabulary[highesttotals[topweights_indices[0:10]]])
#%%
for i in range(n_topics):
    topweights_indices=numpy.argsort(-1*numpy.abs(psi[i,:]))
    print(mycrawled_prunedtexts[topweights_indices[0]][0:100]  ) 
    
    
# topweights_indices=numpy.argsort(-1*numpy.abs(theta[:,1]))
# print(remainingvocabulary[highesttotals[topweights_indices[0:20]]])
# # Same for the next biggest factor (here factor 4)
# topweights_indices=numpy.argsort(-1*numpy.abs(theta[:,4]))
# print(remainingvocabulary[highesttotals[topweights_indices[0:20]]])

#%%6.3

import gensim
# Create a dictionary from the documents
# startdoc=9000 # We will use the baseball n.g.
# enddoc=10000 # again, but all features
gensim_docs=mycrawled_prunedtexts
gensim_dictionary=gensim.corpora.Dictionary(gensim_docs)
# Create the document-term vectors
gensim_docvectors=[]
for k in range(len(gensim_docs)):
    docvector=gensim_dictionary.doc2bow(gensim_docs[k])
    gensim_docvectors.append(docvector)
#%%
# Run the LDA optimization
numtopics=10
randomseed=124574527
numiters=10000
# ninits=10
gensim_ldamodel=gensim.models.ldamodel.LdaModel( \
gensim_docvectors, \
id2word=gensim_dictionary,num_topics=numtopics, \
iterations=numiters,random_state=randomseed)

# Get topic content: term-topic probabilities
gensim_termtopicprobabilities=gensim_ldamodel.get_topics()
# Get topic prevalences per document, and overall topic prevalences
# (expected amount of documents per topic)
overallstrengths=numpy.zeros((numtopics,1))
documentstrengths=numpy.zeros((len(gensim_docvectors),numtopics))
for k in range(len(gensim_docvectors)):
    topicstrengths=gensim_ldamodel.get_document_topics(\
    gensim_docvectors[k],minimum_probability=0)
    for m in range(len(topicstrengths)):
        documentstrengths[k][topicstrengths[m][0]]=topicstrengths[m][1]
        overallstrengths[topicstrengths[m][0]]=\
        overallstrengths[topicstrengths[m][0]]+topicstrengths[m][1]

for i in range(numtopics):
    print(gensim_ldamodel.show_topic(i,topn=10))

for i in range(numtopics):
    topweights_indices= numpy.argsort(-1*numpy.abs(documentstrengths[:,i]))
    print(mycrawled_prunedtexts[topweights_indices[0]][0:100]  ) 
#%%
# Run the LDA optimization
numtopics=15
randomseed=124574527
numiters=10000
# ninits=10
gensim_ldamodel=gensim.models.ldamodel.LdaModel( \
gensim_docvectors, \
id2word=gensim_dictionary,num_topics=numtopics, \
iterations=numiters,random_state=randomseed)

# Get topic content: term-topic probabilities
gensim_termtopicprobabilities=gensim_ldamodel.get_topics()
# Get topic prevalences per document, and overall topic prevalences
# (expected amount of documents per topic)
overallstrengths=numpy.zeros((numtopics,1))
documentstrengths=numpy.zeros((len(gensim_docvectors),numtopics))
for k in range(len(gensim_docvectors)):
    topicstrengths=gensim_ldamodel.get_document_topics(\
    gensim_docvectors[k],minimum_probability=0)
    for m in range(len(topicstrengths)):
        documentstrengths[k][topicstrengths[m][0]]=topicstrengths[m][1]
        overallstrengths[topicstrengths[m][0]]=\
        overallstrengths[topicstrengths[m][0]]+topicstrengths[m][1]

for i in range(numtopics):
    print(gensim_ldamodel.show_topic(i,topn=10))

for i in range(numtopics):
    topweights_indices= numpy.argsort(-1*numpy.abs(documentstrengths[:,i]))
    print(mycrawled_prunedtexts[topweights_indices[0]][0:100]  ) 

























