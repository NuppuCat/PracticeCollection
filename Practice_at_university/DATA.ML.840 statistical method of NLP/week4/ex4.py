# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 07:38:58 2021

@author: One
"""
#%%
temp_fullpath ='D://GRAM//MasterProgramme//Tampere//DATA.ML.840 statistical method of NLP//week4//55-0.txt'

temp_file=open(temp_fullpath,'r',encoding='utf-8',errors='ignore')
wonder=temp_file.read()
temp_file.close()
# print(temp_text)
# downloaded str will change
startstr1 = "[Illustration]"
endstr1 = "*** END OF THE PROJECT GUTENBERG EBOOK THE WONDERFUL WIZARD OF OZ ***"
startind1 = wonder.find(startstr1)
endtind1 = wonder.find(endstr1)

wonder = wonder[(startind1)-2:endtind1]
wonder = wonder.lower()
#%%
import re
import nltk
mytext_paragraphsb=re.split('\n[ \n]*\n', wonder)
mytext_paragraphs=[]
for i in range(13,len(mytext_paragraphsb)):
    if (mytext_paragraphsb[i].startswith("chapter ") or mytext_paragraphsb[i]==''):
       continue
    mytext_paragraphs.append(mytext_paragraphsb[i])
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
for k in range(len(mytext_paragraphs)):
    
    text = nltk.Text(nltk.word_tokenize(mytext_paragraphs[k]))
    lemmatizedtext=lemmatizetext(text)
    lemmatizedtext=nltk.Text(lemmatizedtext)
    mycrawled_lemmatizedtexts.append(lemmatizedtext)
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
# all vocab
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
#%%    
# print(tfidfmatrix.data) 
# Reduce the data to 500 highest-total TF-IDF features
dimensiontotals=numpy.squeeze(numpy.array( \
numpy.sum(tfidfmatrix,axis=0)))
highesttotals=numpy.argsort(-1*dimensiontotals)
Xsmall=tfidfmatrix[:,highesttotals[0:500]]
Xsmall=Xsmall.todense()
# print(Xsmall)

# Normalize the documents to unit vector norm
tempnorms=numpy.squeeze(numpy.array(numpy.sum(numpy.multiply(Xsmall,Xsmall),axis=1)))
# If any documents have zero norm, avoid dividing them by zero
tempnorms[tempnorms==0]=1

td = scipy.sparse.diags(tempnorms**-0.5)
# print(td)


Xsmall=td.dot(Xsmall)
#%%
import sklearn
import sklearn.mixture
# Create the mixture model object, and
# choose the number of components and EM iterations
mixturemodel=sklearn.mixture.GaussianMixture(n_components=10, \
covariance_type='diag',max_iter=100,init_params='random')
fittedmixture=mixturemodel.fit(Xsmall)
sklearn_mixturemodel_means=fittedmixture.means_
sklearn_mixturemodel_weights=fittedmixture.weights_
sklearn_mixturemodel_covariances=fittedmixture.covariances_   

# Find top 20 words with highest mean feature value for each cluster
for k in range(10):
    print(k)
    highest_dimensionweight_indices=numpy.argsort( \
    -numpy.squeeze(sklearn_mixturemodel_means[k,:]),axis=0)
    highest_dimensionweight_indices=highesttotals[highest_dimensionweight_indices]
    print(' '.join(remainingvocabulary[highest_dimensionweight_indices[0:10]]))   


# %% 4.2
lengthofpara = []
for i in range(len(mytext_paragraphs)):
    lengthofpara.append(len(mytext_paragraphs[i]))
# print(lengthofpara)
lengthofpara = numpy.array(lengthofpara)
lind = lengthofpara.argmax()
longestpara = mytext_paragraphs[lind]
print(longestpara)



#%%
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

print('lengthsmooth')
# Use the count statistics to compute the tf-idf matrix
ltfidfmatrix=scipy.sparse.lil_matrix((n_docs,n_vocab))
#length
divm = (numpy.expand_dims(numpy.array(numofterms),axis=1))
divm[divm==0]=1
ltfmatrix = scipy.sparse.lil_matrix(tfmatrix/divm)
# print(tfmatrix)
# print(ltfmatrix.data)
lidfvector= []

    
for k in range(n_vocab):
    lidfvector.append(1+numpy.log(((dfvector[0,k]+1)**-1)*n_docs))
# idfvector=1+numpy.log(((dfvector.data+1)**-1)*n_docs)
idfvector = numpy.array(idfvector)
for k in range(n_docs):
    
     # Combine the tf and idf terms 
     #NOTE HERE SHOULD BE MYLTIPLY, NOT *
     ltfidfmatrix[k,:]=ltfmatrix[k,:].multiply(lidfvector)
     


def printtop20(ltfidfmatrix):
    lhi = []
    
    for k in range(n_vocab):
        lhi.append(ltfidfmatrix[lind,k])
    lhi=numpy.argsort(-1*numpy.array(lhi))
    
    print(' '.join(remainingvocabulary[lhi[:20]]))
printtop20(ltfidfmatrix)
print('logsmooth')
loglltfmatrix=scipy.sparse.lil_matrix((n_docs,n_vocab))
for l in range(n_docs):
    for k in range(n_vocab):
        loglltfmatrix[l,k]=(numpy.log(tfmatrix[l,k]+1))

logtfidfmatrix = scipy.sparse.lil_matrix((n_docs,n_vocab))
for k in range(n_docs):
    
     # Combine the tf and idf terms 
     #NOTE HERE SHOULD BE MYLTIPLY, NOT *
     logtfidfmatrix[k,:]=loglltfmatrix[k,:].multiply(lidfvector)
printtop20(logtfidfmatrix)



print('freqversion')

vmcidfvector = []
alpha = 1/n_vocab
crmfltfmatrix=scipy.sparse.lil_matrix((n_docs,n_vocab))
def sparse_max_row(csr_mat):
    csr_mat=csr_mat.tocsr()
    ret = numpy.maximum.reduceat(csr_mat.data, csr_mat.indptr[:-1])
    ret[numpy.diff(csr_mat.indptr) == 0] = 0
    return ret
docmax = sparse_max_row(tfmatrix)
for l in range(n_docs):
    docmal = docmax[l]
    for k in range(n_vocab):
        num = tfmatrix[l,k]
        crmfltfmatrix[l,k]=alpha+(1-alpha)*num/docmal
for k in range(n_vocab):
    vmcidfvector.append(numpy.log(((dfvector[0,k]+1)**-1)*(n_docs-dfvector[0,k])))
# idfvector=1+numpy.log(((dfvector.data+1)**-1)*n_docs)
vmcidfvector = numpy.array(vmcidfvector)
crmfvcmtfidf  =  scipy.sparse.lil_matrix((n_docs,n_vocab))
for k in range(n_docs):
    
     # Combine the tf and idf terms 
     #NOTE HERE SHOULD BE MYLTIPLY, NOT *
     crmfvcmtfidf[k,:]=crmfltfmatrix[k,:].multiply(vmcidfvector)
printtop20(crmfvcmtfidf)     

# freqlltfmatrix = numpy.array(freqlltfmatrix)
# vmcidfvector = numpy.array(vmcidfvector)
# freqversiontfidfmatrix = freqlltfmatrix.multiply(vmcidfvector)
# gettopnwords(10,20,freqversiontfidfmatrix)

#%%    
# # print(tfidfmatrix.data) 
# # Reduce the data to 500 highest-total TF-IDF features
# dimensiontotals=numpy.squeeze(numpy.array( \
# numpy.sum(tfidfmatrix,axis=0)))
# highesttotals=numpy.argsort(-1*dimensiontotals)













# #%%
# def gettopnwords(n_components,n,tfidfmatrix):
#     tfidfmatrix = tfidfmatrix.tolil()
#     dimensiontotals=numpy.squeeze(numpy.array( numpy.sum(tfidfmatrix,axis=0)))
#     highesttotals=numpy.argsort(-1*dimensiontotals)
#     Xsmall=tfidfmatrix[:,highesttotals[0:500]]
#     Xsmall=Xsmall.todense()
#     # print(Xsmall)
    
#     # Normalize the documents to unit vector norm
#     tempnorms=numpy.squeeze(numpy.array(numpy.sum(numpy.multiply(Xsmall,Xsmall),axis=1)))
#     # If any documents have zero norm, avoid dividing them by zero
#     tempnorms[tempnorms==0]=1
    
#     # print(td)
    
    
#     Xsmall=scipy.sparse.diags(tempnorms**-0.5).dot(Xsmall)
    
#     # Create the mixture model object, and
#     # choose the number of components and EM iterations
    
#     mixturemodel=sklearn.mixture.GaussianMixture(n_components=n_components, \
#     covariance_type='diag',max_iter=100,init_params='random')
#     fittedmixture=mixturemodel.fit(Xsmall)
#     sklearn_mixturemodel_means=fittedmixture.means_
#     sklearn_mixturemodel_weights=fittedmixture.weights_
#     sklearn_mixturemodel_covariances=fittedmixture.covariances_   
    
#     # Find top 20 words with highest mean feature value for each cluster
#     for k in range(n_components):
#         print(k)
#         highest_dimensionweight_indices=numpy.argsort( \
#         -numpy.squeeze(sklearn_mixturemodel_means[k,:]),axis=0)
#         highest_dimensionweight_indices=highesttotals[highest_dimensionweight_indices]
#         print(' '.join(remainingvocabulary[highest_dimensionweight_indices[0:n]]))   



# # print(len(longestpara)==numpy.max(lengthofpara))
# ltext = nltk.Text(nltk.word_tokenize(longestpara))
# llemmatizedtext=lemmatizetext(ltext)
# llemmatizedtext=nltk.Text(llemmatizedtext)
# uniqueresults=numpy.unique(llemmatizedtext,return_inverse=True)
# longestparalemtext=uniqueresults[0]
# lmyindices_in_unifiedvocabulary=uniqueresults[1]

# pruningdecisions=numpy.zeros((len(longestparalemtext),1))
# for k in range(len(longestparalemtext)):
#     # Rule 1: check the nltk stop word list
#     if (unifiedvocabulary[k] in nltkstopwords):
#         pruningdecisions[k]=1
#     # Rule 2: if the word is in the top 1% of frequent words
#     if (k in highest_totaloccurrences_indices[\
#         0:int(numpy.floor(len(unifiedvocabulary)*0.01))]):
#         pruningdecisions[k]=1
#     # # Rule 3: if the word is in the bottom 65% of frequent words
#     # if (k in highest_totaloccurrences_indices[(int(numpy.floor(\
#     #     len(unifiedvocabulary)*0.35))):len(unifiedvocabulary)]):
#     #     pruningdecisions[k]=1
#     # Rule 4: if the word is too short
#     if len(unifiedvocabulary[k])<2:
#         pruningdecisions[k]=1
#     # Rule 5: if the word is too long
#     if len(unifiedvocabulary[k])>20:
#         pruningdecisions[k]=1
#     # Rule 6: if the word has unwanted characters
#     # (here for simplicity only a-z allowed)
#     # if unifiedvocabulary[k].isalpha()==False:
#     #     pruningdecisions[k]=1
#     # Rule 7, if the word courred less than 4 times.
#     if (unifiedvocabulary_totaloccurrencecounts[k]<4):
#         pruningdecisions[k]=1

# lremainingvocabulary = longestparalemtext[numpy.squeeze(pruningdecisions!=1)]



# ln_docs=n_docs
# ln_vocab=len(lremainingvocabulary)
# # Matrix of term frequencies
# ltfmatrix=scipy.sparse.lil_matrix((ln_docs,ln_vocab))
# # Row vector of document frequencies
# ldfvector=scipy.sparse.lil_matrix((1,ln_vocab))
# # Loop over documents

# # Row vector of which words occurred in this document
# ltemp_dfvector=scipy.sparse.lil_matrix((1,ln_vocab))
# # Loop over words
# for l in range(len(longestparalemtext)):
#     # Add current word to term-frequency count and document-count
#     currentword=lmyindices_in_unifiedvocabulary[l]
#     ltfmatrix[0,currentword]=ltfmatrix[0,currentword]+1
#     ltemp_dfvector[0,currentword]=1


# # Add which words occurred in this document to overall document counts
# ldfvector=ldfvector+ltemp_dfvector


# #length
# lnumofterms = len(llemmatizedtext)

# # Use the count statistics to compute the tf-idf matrix
# ltfidfmatrix=scipy.sparse.lil_matrix((ln_docs,ln_vocab))
# #length
# lltfmatrix = scipy.sparse.lil_matrix(ltfmatrix/lnumofterms)
# # print(tfmatrix)
# # print(ltfmatrix.data)
# lidfvector = []
# for k in range(ln_vocab):
#     lidfvector.append(1+numpy.log(((ldfvector[0,k]+1)**-1)*ln_docs))
# # lidfvector=1+numpy.log(((ldfvector.data+1)**-1)*ln_docs)
# lidfvector = numpy.array(lidfvector)
# ltfidfmatrix=lltfmatrix.multiply(lidfvector)

# print('lengthsmooth')
# gettopnwords(10,20,ltfidfmatrix)
# print('logsmooth')
# loglltfmatrix=[]
# for k in range(ln_vocab):
#     loglltfmatrix.append(numpy.log(((ltfmatrix[0,k]+1))))
# loglltfmatrix=numpy.array(loglltfmatrix)
# logtfidfmatrix = scipy.sparse.lil_matrix((ln_docs,ln_vocab))
# logtfidfmatrix = loglltfmatrix.multiply(lidfvector)
# gettopnwords(10,20,logtfidfmatrix)
# print('freqversion')
# freqlltfmatrix=[]
# vmcidfvector = []
# alpha = 1/ln_vocab
# for k in range(ln_vocab):
#     moustfreqwordcount = ltfmatrix.data.max
#     freqlltfmatrix.append(alpha+(1-alpha)*ltfmatrix[0,k]/moustfreqwordcount)
#     # bcs only one doc
#     vmcidfvector.append(numpy.log(1/(1+ldfvector[0,k])))
# freqlltfmatrix = numpy.array(freqlltfmatrix)
# vmcidfvector = numpy.array(vmcidfvector)
# freqversiontfidfmatrix = freqlltfmatrix.multiply(vmcidfvector)
# gettopnwords(10,20,freqversiontfidfmatrix)

#%% 4.3



# longestparalemtext = mycrawled_prunedtexts[lengthofpara.argmax()]
# longestmyindices_in_prunedvocabulary = myindices_in_prunedvocabulary[lengthofpara.argmax()]
# print(longestparalemtext[0])
#%%
# def pruntext(unifiedvocabulary,mycrawled_lemmatizedtexts,myindices_in_unifiedvocabulary):
#     nltk.download('stopwords')
#     nltkstopwords=nltk.corpus.stopwords.words('english')
#     pruningdecisions=numpy.zeros((len(unifiedvocabulary),1))
#     for k in range(len(unifiedvocabulary)):
#         # Rule 1: check the nltk stop word list
#         if (unifiedvocabulary[k] in nltkstopwords):
#             pruningdecisions[k]=1
#         # Rule 2: if the word is in the top 1% of frequent words
#         if (k in highest_totaloccurrences_indices[\
#             0:int(numpy.floor(len(unifiedvocabulary)*0.01))]):
#             pruningdecisions[k]=1
#         # # Rule 3: if the word is in the bottom 65% of frequent words
#         # if (k in highest_totaloccurrences_indices[(int(numpy.floor(\
#         #     len(unifiedvocabulary)*0.35))):len(unifiedvocabulary)]):
#         #     pruningdecisions[k]=1
#         # Rule 4: if the word is too short
#         if len(unifiedvocabulary[k])<2:
#             pruningdecisions[k]=1
#         # Rule 5: if the word is too long
#         if len(unifiedvocabulary[k])>20:
#             pruningdecisions[k]=1
#         # Rule 6: if the word has unwanted characters
#         # (here for simplicity only a-z allowed)
#         # if unifiedvocabulary[k].isalpha()==False:
#         #     pruningdecisions[k]=1
#         # Rule 7, if the word courred less than 4 times.
#         if (unifiedvocabulary_totaloccurrencecounts[k]<4):
#             pruningdecisions[k]=1
#     # pruned_highest_totaloccurrences_indices = numpy.argsort(numpy.multiply(-1*unifiedvocabulary_totaloccurrencecounts,(pruningdecisions!=1)),axis=0)
#     # squeeze:Remove axes of length one from a.
#     # print(numpy.squeeze(unifiedvocabulary[pruned_highest_totaloccurrences_indices[0:100]]))
#     # print(numpy.squeeze(unifiedvocabulary_totaloccurrencecounts[pruned_highest_totaloccurrences_indices[0:100]]))
#     # astemp= -1*unifiedvocabulary_totaloccurrencecounts*pruningdecisions
    
#     oldtopruned=[]
#     tempind=-1
#     remainingvocabulary = unifiedvocabulary[numpy.squeeze(pruningdecisions!=1)]
#     for k in range(len(unifiedvocabulary)):
#         if pruningdecisions[k]==0:
#             tempind=tempind+1
#             oldtopruned.append(tempind)
#         else:
#             oldtopruned.append(-1)
#     #Create pruned texts
#     mycrawled_prunedtexts=[]
#     myindices_in_prunedvocabulary=[]
#     for k in range(len(mycrawled_lemmatizedtexts)):
#         # print(k)
#         temp_newindices=[]
#         temp_newdoc=[]
#         for l in range(len(mycrawled_lemmatizedtexts[k])):
#             temp_oldindex=myindices_in_unifiedvocabulary[k][l]
#             temp_newindex=oldtopruned[temp_oldindex]
#             if temp_newindex!=-1:
#                 temp_newindices.append(temp_newindex)
#                 temp_newdoc.append(unifiedvocabulary[temp_oldindex])
#         mycrawled_prunedtexts.append(temp_newdoc)
#         myindices_in_prunedvocabulary.append(temp_newindices)
#     return (remainingvocabulary,mycrawled_prunedtexts,myindices_in_prunedvocabulary)
# remainingvocabulary,mycrawled_prunedtexts,myindices_in_prunedvocabulary = pruntext(longestparalemtext,llemmatizedtext,lmyindices_in_unifiedvocabulary)
#%%
# import scipy
# ln_docs=1
# ln_vocab=len(longestparalemtext)
# # Matrix of term frequencies
# ltfmatrix=scipy.sparse.lil_matrix((ln_docs,ln_vocab))
# # Row vector of document frequencies
# ldfvector=scipy.sparse.lil_matrix((1,ln_vocab))
# # Loop over documents

# # Row vector of which words occurred in this document
# ltemp_dfvector=scipy.sparse.lil_matrix((1,ln_vocab))
# # Loop over words
# for l in range(len(longestparalemtext)):
#     # Add current word to term-frequency count and document-count
#     currentword=lmyindices_in_unifiedvocabulary[l]
#     ltfmatrix[0,currentword]=ltfmatrix[0,currentword]+1
#     ltemp_dfvector[0,currentword]=1


# # Add which words occurred in this document to overall document counts
# ldfvector=ldfvector+ltemp_dfvector


# #length
# lnumofterms = len(llemmatizedtext)

# # Use the count statistics to compute the tf-idf matrix
# ltfidfmatrix=scipy.sparse.lil_matrix((ln_docs,ln_vocab))
# #length
# lltfmatrix = scipy.sparse.lil_matrix(ltfmatrix/lnumofterms)
# # print(tfmatrix)
# # print(ltfmatrix.data)
# lidfvector = []
# for k in range(ln_vocab):
#     lidfvector.append(1+numpy.log(((ldfvector[0,k]+1)**-1)*ln_docs))
# # lidfvector=1+numpy.log(((ldfvector.data+1)**-1)*ln_docs)
# lidfvector = numpy.array(lidfvector)
# ltfidfmatrix=lltfmatrix.multiply(lidfvector)

# def gettopnwords(n_components,n,tfidfmatrix):
#     tfidfmatrix = tfidfmatrix.tolil()
#     dimensiontotals=numpy.squeeze(numpy.array( numpy.sum(tfidfmatrix,axis=0)))
#     highesttotals=numpy.argsort(-1*dimensiontotals)
#     Xsmall=tfidfmatrix[:,highesttotals[0:500]]
#     Xsmall=Xsmall.todense()
#     # print(Xsmall)
    
#     # Normalize the documents to unit vector norm
#     tempnorms=numpy.squeeze(numpy.array(numpy.sum(numpy.multiply(Xsmall,Xsmall),axis=1)))
#     # If any documents have zero norm, avoid dividing them by zero
#     tempnorms[tempnorms==0]=1
    
#     # print(td)
    
#     try:
#         Xsmall=scipy.sparse.diags(tempnorms**-0.5).dot(Xsmall)
#     except:
#         Xsmall=scipy.sparse.diags((numpy.expand_dims(numpy.array(tempnorms),axis=0))**-0.5).dot(Xsmall)
#         Xsmall = numpy.squeeze(Xsmall)
#         Xsmall=Xsmall.T
#     # Create the mixture model object, and
#     # choose the number of components and EM iterations
    
#     mixturemodel=sklearn.mixture.GaussianMixture(n_components=n_components, \
#     covariance_type='diag',max_iter=100,init_params='random')
#     fittedmixture=mixturemodel.fit(Xsmall)
#     sklearn_mixturemodel_means=fittedmixture.means_
#     sklearn_mixturemodel_weights=fittedmixture.weights_
#     sklearn_mixturemodel_covariances=fittedmixture.covariances_   
    
#     # Find top 20 words with highest mean feature value for each cluster
#     for k in range(n_components):
#         print(k)
#         highest_dimensionweight_indices=numpy.argsort( \
#         -numpy.squeeze(sklearn_mixturemodel_means[k,:]),axis=0)
#         highest_dimensionweight_indices=highesttotals[highest_dimensionweight_indices]
#         print(' '.join(remainingvocabulary[highest_dimensionweight_indices[0:n]]))   

# print('lengthsmooth')
# gettopnwords(10,20,ltfidfmatrix)
# print('logsmooth')
# loglltfmatrix=[]
# for k in range(ln_vocab):
#     loglltfmatrix.append(numpy.log(((ltfmatrix[0,k]+1))))
# loglltfmatrix=numpy.array(loglltfmatrix)
# logtfidfmatrix = scipy.sparse.lil_matrix((ln_docs,ln_vocab))
# logtfidfmatrix = loglltfmatrix.multiply(lidfvector)
# gettopnwords(10,20,logtfidfmatrix)
# print('freqversion')
# freqlltfmatrix=[]
# vmcidfvector = []
# alpha = 1/ln_vocab
# for k in range(ln_vocab):
#     moustfreqwordcount = ltfmatrix.data.max
#     freqlltfmatrix.append(alpha+(1-alpha)*ltfmatrix[0,k]/moustfreqwordcount)
#     # bcs only one doc
#     vmcidfvector.append(numpy.log(1/(1+ldfvector[0,k])))
# freqlltfmatrix = numpy.array(freqlltfmatrix)
# vmcidfvector = numpy.array(vmcidfvector)
# freqversiontfidfmatrix = freqlltfmatrix.multiply(vmcidfvector)
# gettopnwords(10,20,freqversiontfidfmatrix)


#%%






# #%% Use the TF-IDF matrix as data to be clustered
# X=tfidfmatrix
# # Normalize the documents to unit vector norm
# tempnorms=numpy.squeeze(numpy.array(numpy.sum(X.multiply(X),axis=1)))
# # If any documents have zero norm, avoid dividing them by zero
# tempnorms[tempnorms==0]=1
# X=scipy.sparse.diags(tempnorms**-0.5).dot(X)
# n_data=numpy.shape(X)[0]
# n_dimensions=numpy.shape(X)[1]
# # print(X.data)
# #%% Initialize the Gaussian mixture model
# # Function to initialize the Gaussian mixture model, create component parameters
# def initialize_mixturemodel(X,n_components):
#     # Create lists of sparse matrices to hold the parameters
#     n_dimensions=numpy.shape(X)[1]
#     mixturemodel_means=scipy.sparse.lil_matrix((n_components,n_dimensions))
#     mixturemodel_weights=numpy.zeros((n_components))
#     mixturemodel_covariances=[]
#     mixturemodel_inversecovariances=[]
#     for k in range(n_components):
#         tempcovariance=scipy.sparse.lil_matrix((n_dimensions,n_dimensions))
#         mixturemodel_covariances.append(tempcovariance)
#         tempinvcovariance=scipy.sparse.lil_matrix((n_dimensions,n_dimensions))
#         mixturemodel_inversecovariances.append(tempinvcovariance)
#     # Initialize the parameters
#         mixturemodel_weights[k]=1/n_components
#         # Pick a random data point as the initial mean
#         tempindex=scipy.stats.randint.rvs(low=0,high=n_data)
#         mixturemodel_means[k]=X[tempindex,:].toarray()
#     # Initialize the covariance matrix to be spherical
#     for l in range(n_dimensions):
#         mixturemodel_covariances[k][l,l]=1
#         mixturemodel_inversecovariances[k][l,l]=1
#     # print(mixturemodel_weights)
#     return(mixturemodel_weights,mixturemodel_means,mixturemodel_covariances,\
#     mixturemodel_inversecovariances)    
# # Added  n_components = 10 and import
# import numpy.matlib
# def run_estep(X,mixturemodel_means,mixturemodel_covariances, \
#     mixturemodel_inversecovariances,mixturemodel_weights,n_components = 10):
#     # For each component, compute terms that do not involve data
#     meanterms=numpy.zeros((n_components))
#     logdeterminants=numpy.zeros((n_components))
#     logconstantterms=numpy.zeros((n_components))
#     for k in range(n_components):
#         # Compute mu_k*inv(Sigma_k)*mu_k
#         meanterms[k]=(mixturemodel_means[k,:].dot( \
#         mixturemodel_inversecovariances[k]).dot(mixturemodel_means[k,:].T))[0,0]
#         # Compute determinant of Sigma_k. For a diagonal matrix
#         # this is just the product of the main diagonal
#         logdeterminants[k]=numpy.sum(numpy.log(mixturemodel_covariances[k].diagonal(0)))
#         # Compute constant term beta_k * 1/(|Sigma_k|^1/2)
#         # Omit the (2pi)^d/2 as it cancels out
#         logconstantterms[k]=numpy.log(mixturemodel_weights[k]) - 0.5*logdeterminants[k]
    
#     print('E-step part2 ')
#     # Compute terms that involve distances of data from components
#     xnorms=numpy.zeros((n_data,n_components))
#     xtimesmu=numpy.zeros((n_data,n_components))
#     for k in range(n_components):
#         # print(k)
#         xnorms[:,k]=(X.dot(mixturemodel_inversecovariances[k]).dot(X.T)).diagonal(0)
#         xtimesmu[:,k]=numpy.squeeze((X.dot(mixturemodel_inversecovariances[k]).dot( \
#         mixturemodel_means[k,:].T)).toarray())
#     xdists=xnorms+numpy.matlib.repmat(meanterms,n_data,1)-2*xtimesmu
#     # Substract maximal term before exponent (cancels out) to maintain computational precision
#     numeratorterms=logconstantterms-xdists/2
#     numeratorterms-=numpy.matlib.repmat(numpy.max(numeratorterms,axis=1),n_components,1).T
#     numeratorterms=numpy.exp(numeratorterms)
#     mixturemodel_componentmemberships=numeratorterms/numpy.matlib.repmat( \
#     numpy.sum(numeratorterms,axis=1),n_components,1).T
#     return(mixturemodel_componentmemberships)
# def run_mstep_sumweights(mixturemodel_componentmemberships):
#     # Compute total weight per component
#     mixturemodel_weights=numpy.sum(mixturemodel_componentmemberships,axis=0)
#     return(mixturemodel_weights)
# def run_mstep_means(X,mixturemodel_componentmemberships,mixturemodel_weights):
#     # Update component means
#     mixturemodel_means=scipy.sparse.lil_matrix((n_components,n_dimensions))
#     for k in range(n_components):
#         mixturemodel_means[k,:]=\
#         numpy.sum(scipy.sparse.diags(mixturemodel_componentmemberships[:,k]).dot(X),axis=0)
#         mixturemodel_means[k,:]/=mixturemodel_weights[k]
#     return(mixturemodel_means)
# def run_mstep_covariances(X,mixturemodel_componentmemberships,mixturemodel_weights,mixturemodel_means):
#     # Update diagonal component covariance matrices
#     n_dimensions=numpy.shape(X)[1]
#     n_components=numpy.shape(mixturemodel_componentmemberships)[1]
#     tempcovariances=numpy.zeros((n_components,n_dimensions))
#     mixturemodel_covariances=[]
#     mixturemodel_inversecovariances=[]
#     for k in range(n_components):
#         tempcovariances[k,:]= \
#         numpy.sum(scipy.sparse.diags(mixturemodel_componentmemberships[:,k]).dot(X.multiply(X)),axis=0) \
#         -mixturemodel_means[k,:].multiply(mixturemodel_means[k,:])*mixturemodel_weights[k]
#         tempcovariances[k,:]/=mixturemodel_weights[k]
#         # Convert to sparse matrices
#         tempepsilon=1e-10
#         # Add a small regularization term
#         temp_covariance=scipy.sparse.diags(tempcovariances[k,:]+tempepsilon)
#         temp_inversecovariance=scipy.sparse.diags((tempcovariances[k,:]+tempepsilon)**-1)
#         mixturemodel_covariances.append(temp_covariance)
#         mixturemodel_inversecovariances.append(temp_inversecovariance)
#     return(mixturemodel_covariances,mixturemodel_inversecovariances)
# def run_mstep_normalizeweights(mixturemodel_weights):
#     # Update mixture-component prior probabilities
#     mixturemodel_weights/=sum(mixturemodel_weights)
#     return(mixturemodel_weights)
        
# #%% Perform the EM algorithm iterations
# def perform_emalgorithm(X,n_components,n_emiterations):
#     mixturemodel_weights,mixturemodel_means,mixturemodel_covariances,\
#     mixturemodel_inversecovariances=initialize_mixturemodel(X,n_components)
#     for t in range(n_emiterations):
#         # ====== E-step: Compute the component membership
#         # probabilities of each data point ======
#         print('E-step ' + str(t))

#         mixturemodel_componentmemberships=run_estep(X,mixturemodel_means,mixturemodel_covariances,\
#         mixturemodel_inversecovariances,mixturemodel_weights,n_components)
#         # ====== M-step: update component parameters======
#         print('M-step ' + str(t))
#         print('M-step part1 ' + str(t))
#         mixturemodel_weights=run_mstep_sumweights(mixturemodel_componentmemberships)
#         print('M-step part2 ' + str(t))
#         mixturemodel_means=run_mstep_means(X,mixturemodel_componentmemberships,mixturemodel_weights)
#         print('M-step part3 ' + str(t))
#         mixturemodel_covariances,mixturemodel_inversecovariances=run_mstep_covariances(X,\
#         mixturemodel_componentmemberships,mixturemodel_weights,mixturemodel_means)
#         print('M-step part4 ' + str(t))
#         mixturemodel_weights=run_mstep_normalizeweights(mixturemodel_weights)
#         return(mixturemodel_weights,mixturemodel_means,mixturemodel_covariances,\
#         mixturemodel_inversecovariances)
# # Try out the functions we just defined on the data
# n_components=10
# n_emiterations=1000
# mixturemodel_weights,mixturemodel_means,mixturemodel_covariances,\
# mixturemodel_inversecovariances = perform_emalgorithm(X,n_components,n_emiterations)    
# # print(X.data)   
    
# # Find top 20 words for each cluster
# for k in range(n_components):
#     print(k)
#     highest_dimensionweight_indices=\
#     numpy.argsort(-numpy.squeeze(\
#     mixturemodel_means[k,:].toarray()),axis=0)
#     print(' '.join(remainingvocabulary[\
#     highest_dimensionweight_indices[1:10]]))    
    
    
    
    
    
    
    