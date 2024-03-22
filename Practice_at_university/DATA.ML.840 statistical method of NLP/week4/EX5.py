# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:39:48 2021

@author: One
"""

temp_fullpath ='D://GRAM//MasterProgramme//Tampere//DATA.ML.840 statistical method of NLP//week5&6//pg23731.txt'

temp_file=open(temp_fullpath,'r',encoding='utf-8',errors='ignore')
odyssey=temp_file.read()
temp_file.close()
# print(temp_text)
# downloaded str will change
startstr1 = "\nA MARTIAN ODYSSEY\n"


endstr1 = "*** END OF THIS PROJECT GUTENBERG EBOOK A MARTIAN ODYSSEY ***"
startind1 = odyssey.find(startstr1)+len(startstr1)+2
endtind1 = odyssey.find(endstr1)

odyssey = odyssey[(startind1)-2:endtind1]
odyssey = odyssey.lower()
# print(odyssey)
temp_fullpath ='D://GRAM//MasterProgramme//Tampere//DATA.ML.840 statistical method of NLP//week5&6//pg10148.txt'

temp_file=open(temp_fullpath,'r',encoding='utf-8',errors='ignore')
robin =temp_file.read()
temp_file.close()
# print(temp_text)
# downloaded str will change
startstr1 = " START OF THIS PROJECT GUTENBERG EBOOK THE MERRY ADVENTURES OF ROBIN HOOD ***"
endstr1 = "*** END OF THIS PROJECT GUTENBERG EBOOK THE MERRY ADVENTURES OF ROBIN HOOD ***"
startind1 = robin.find(startstr1)+len(startstr1)+2
endtind1 = robin.find(endstr1)

robin = robin[(startind1)-2:endtind1]
robin = robin.lower()
# print(robin)
import re
import nltk
import numpy
odysseytext = nltk.Text(nltk.word_tokenize(odyssey))
robintext = nltk.Text(nltk.word_tokenize(robin))

odysseyuniqueresults=numpy.unique(odysseytext,return_inverse=True)
odysseyvocabularies=odysseyuniqueresults[0]
odysseyindices_in_vocabularies=odysseyuniqueresults[1]

robinuniqueresults=numpy.unique(robintext,return_inverse=True)
robinvocabularies=robinuniqueresults[0]
robinindices_in_vocabularies=robinuniqueresults[1]
#%%
# import numpy
# import scipy
# def build_ngram(textindexvector,n_vocab,maxN):
#     # Create the overall structure that will store the n-gram
#     allprobstructs_NextCount=[]
#     allprobstructs_NextProb=[]
#     allprobstructs_PreviousStruct=[]
#     # Create unigram probability table, store it as the root of probtables
#     tempstruct_NextCount=scipy.sparse.dok_matrix((n_vocab,1))
#     tempstruct_NextProb=scipy.sparse.dok_matrix((n_vocab,1))
#     tempstruct_PreviousStruct=scipy.sparse.dok_matrix((n_vocab,1))
#     allprobstructs_NextCount.append(tempstruct_NextCount)
#     allprobstructs_NextProb.append(tempstruct_NextProb)
#     allprobstructs_PreviousStruct.append(tempstruct_PreviousStruct)
#     nstructs=1
#     # Count how many probability tables have been created at different
#     # n-gram levels. Because this is a zero-based index, the index of the
#     # level indicating how long a history each n-gram takes into account:
#     # 0 for unigrams, 1 for bigrams, and so on.
#     nstructsperlevel=numpy.zeros((maxN))
#     # Initially there is only one table which is a unigram-table.
#     nstructsperlevel[0]=1
    
#     # Iterate through the text
#     for t in range(len(textindexvector)):
#         if (t%10)==0:
#             # print(str(t) + ' ' + str(nstructsperlevel))
#             1==1
#         # Vocabulary index of the word at position t in the text
#         tempword=textindexvector[t-0];
#         # Suppose we have words w(1),...,w(t-3),w(t-2),w(t-1),w(t) in the text.
#         # Then the transition to w(t) must be recorded for several n-grams:
#         # []-->w(t) : unigram transition (n=1)
#         # [w(t-1)]-->w(t) : bigram transition (n=2)
#         # [w(t-2),w(t-1)]-->w(t) : trigram transition (n=3)
#         # [w(t-3),w(t-2),w(t-1)]-->w(t) : 4-gram transition (n=4)
#         # [w(t-4),w(t-3),w(t-2),w(t-1)]-->w(t) : 5-gram transition (n=5)
#         # Start from the unigram (root of the tables), record the transition
#         currentstruct=0
#         # Record the transition into the transition counts:
#         # and its count in the unigram model increases by 1.
#         #allprobstructs[currentstruct]['Next'][tempword,0]=1
#         allprobstructs_NextCount[currentstruct][tempword,0]+=1
#         # Now record this transition into higher-level n-grams.
#         # Address history up to maximum N-gram length or beginning of
#         # the text, whichever is sooner.
#         # Iterate a zero-based index "n" of steps back
#         for n in range(min([maxN-1,t])):
#         # Take the next step back.
#         # Vocabulary index of the previous word
#             previousword=textindexvector[t-n-1];
#             # At this point in the for loop, the current probability table
#             # allprobstructs[currentstruct] represents a (n+1)-gram which uses
#             # a history of length (n): "[w(t-n),...,w(t-1)]--->NextWord".
#             # The "previousword" w(t-n-1) is an expansion of it into a
#             # (n+2)-gram which uses a history of length (n+1):
#             # "[w(t-n-1),...,w(t-1)]--->NextWord". This expansion might exist
#             # already or not. The field 'Previous' of the current (n+1)-gram
#             # records whether that expansion exists.
#             # Create a new history reference (next-level ngram) if it
#             # did not exist.
#             # Note that the unigram table has index 0, but it is never an
#             # expansion of a smaller n-gram.
#             if allprobstructs_PreviousStruct[currentstruct][previousword,0]==0:
#             # Create the probability table for the expansion. Because this
#             # is the first time this "[w(t-n-1),...,w(t-1)]--->NextWord" has
#             # been needed, initially it has no next-words (the current
#             # word will become its first observed next-word). Similarly,
#             # because this is the first time this table is needed, it cannot
#             # have any higher-level expansions yet.

#                 tempstruct_NextCount=scipy.sparse.dok_matrix((n_vocab,1));
#                 tempstruct_NextProb=scipy.sparse.dok_matrix((n_vocab,1));
#                 tempstruct_PreviousStruct=scipy.sparse.dok_matrix((n_vocab,1));
#                 # Add the created table into the overall list of all probability
#                 # tables, increase the count of tables overall and at the n-gram
#                 # level (history length n+1) where the table was created.
#                 nstructs+=1;
#                 nstructsperlevel[n+1]+=1;
#                 allprobstructs_NextCount.append(tempstruct_NextCount)
#                 allprobstructs_NextProb.append(tempstruct_NextProb)
#                 allprobstructs_PreviousStruct.append(tempstruct_PreviousStruct)
#                 # Mark that the expansion now exists into the current
#                 # current "[w(t-n),...,w(t-1)]--->NextWord" table
#                 # allprobstructs[currentstruct]['Previous'][previousword,0]=1
#                 # Add a pointer from the current table to the newly
#                 # created structure (index of the newly created table in the
#                 # overall list)
#                 allprobstructs_PreviousStruct \
#                 [currentstruct][previousword,0]=nstructs-1;
#                 # At this point we can be sure the next-level n-gram exists, so we
#                 # go to the next-level ngram and add the newest word-occurrence to it
#                 # as a possible next word, increasing its count.
#                 currentstruct=allprobstructs_PreviousStruct \
#                 [currentstruct][previousword,0];
#                 currentstruct=int(currentstruct)
#                 allprobstructs_NextCount[currentstruct][tempword,0]+=1
#     # For all tables that have been created, obtain their probabilities by
#     # normalizing their counts
#     for k in range(nstructs):
#         allprobstructs_NextProb[k]=allprobstructs_NextCount[k] \
#         /numpy.sum(allprobstructs_NextCount[k]);
#     return((allprobstructs_NextCount,allprobstructs_NextProb,allprobstructs_PreviousStruct))

# import numpy
# import scipy
# import scipy.stats
# def sample_ngram(allprobstructs,n_words_to_sample,maxN,initialtext):
#     allprobstructs_NextProb=allprobstructs[1]
#     allprobstructs_PreviousStruct=allprobstructs[2]
#     sampletext=[]
#     if len(initialtext)!=0:
#         for t in initialtext:
#             # print(t[0])
#             sampletext.append(t[0][0])
#     # sampletext.extend(initialtext)
#     for k in range(n_words_to_sample):
#         # We are sampling a new word for position t
#         t=len(initialtext)+k
#         # Start from unigram probability table
#         currentstruct=0
#         # Try to use as much history as possible for sampling the next
#         # word, but revert to smaller n-gram if data is not available for
#         # the current history
#         historycount=len(initialtext)
#         atempv = min(maxN-1,t)
#         # print(atempv)
#         for n in range(atempv):
#             # If we want, we can set a probability to use a higher-level n-gram
#             usehigherlevel_probability=0.99
#             if (scipy.stats.uniform.rvs() < usehigherlevel_probability):
#                 # Try to advance to the next-level n-gram
#                 historycount=historycount+1
#                 # print((t,historycount,len(sampletext)))
#                 previousword=sampletext[t-historycount]
#                 # print(previousword)
#                 if allprobstructs_PreviousStruct[currentstruct][previousword,0]>0:
#                     currentstruct=allprobstructs_PreviousStruct[currentstruct][previousword,0]
#                     currentstruct=int(currentstruct)
#                 else:
#                 # Don't try to advance any more times, just exist the for-loop
#                     break
#         # At this point we have found a probability table at some history level.
#         # Sample from its nonzero entries.
#         possiblewords=allprobstructs_NextProb[currentstruct].nonzero()[0]
        
#         possibleprobs=numpy.squeeze(allprobstructs_NextProb[currentstruct][possiblewords,0].toarray(),axis=1)
#         print(possiblewords)
#         print(possibleprobs)
#         currentword=numpy.random.choice(possiblewords, p=possibleprobs)
        
#         sampletext.append(currentword)
#     # Return the created text
#     return(sampletext)
# #%%
# maxN=[1,2,3,5]
# for i in maxN:
#     print('maxN is: ',i)
#     odysseyngram=build_ngram(odysseyindices_in_vocabularies,len(odysseyvocabularies),i)
#     # This can be an array of vocabulary indices of previously observed words
#     initialtext=[]
#     # Sample a vector of word indices from the 5-gram
#     # following the initial text
#     n_words_to_sample=100
#     sampledtext=sample_ngram(odysseyngram,n_words_to_sample,i,initialtext)
#     # Print the result
#     print(' '.join(odysseyvocabularies[sampledtext]))
   
# for i in maxN:
#     print('maxN is: ',i)
#     robinngram=build_ngram(robinindices_in_vocabularies,len(robinvocabularies),i)
#     # This can be an array of vocabulary indices of previously observed words
#     initialtext=[]
#     # Sample a vector of word indices from the 5-gram
#     # following the initial text
#     n_words_to_sample=100
#     sampledtext=sample_ngram(robinngram,n_words_to_sample,i,initialtext)
#     # Print the result
#     print(' '.join(robinvocabularies[sampledtext]))
# #%% 
# maxN=[2,3]
# initialtext=['the','moon']
# od1 = numpy.where(odysseyvocabularies=='the')
# od2 = numpy.where(odysseyvocabularies=='moon')
# rob1 = numpy.where(robinvocabularies=='the')
# rob2 = numpy.where(robinvocabularies=='moon')
# #%%
# for i in maxN:
#     print('maxN is: ',i)
#     odysseyngram=build_ngram(odysseyindices_in_vocabularies,len(odysseyvocabularies),i)
#     # This can be an array of vocabulary indices of previously observed words
#     initialtextod = [od1,od2]
#     # Sample a vector of word indices from the 5-gram
#     # following the initial text
#     n_words_to_sample=100
#     sampledtext=sample_ngram(odysseyngram,n_words_to_sample,i,initialtextod)
#     # Print the result
#     print(' '.join(odysseyvocabularies[sampledtext]))

# for i in maxN:
#     print('maxN is: ',i)
#     robinngram=build_ngram(robinindices_in_vocabularies,len(robinvocabularies),i)
#     # This can be an array of vocabulary indices of previously observed words
#     initialtextrob = [rob1,rob2]
#     # Sample a vector of word indices from the 5-gram
#     # following the initial text
#     n_words_to_sample=100
#     sampledtext=sample_ngram(robinngram,n_words_to_sample,i,initialtextrob)
#     # Print the result
#     print(' '.join(robinvocabularies[sampledtext]))










#%%
import nltk
import nltk.lm
# Create some example text documents as lists of their words
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk import word_tokenize, sent_tokenize 
# Create N-gram training data

from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenize = TreebankWordDetokenizer().detokenize

def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)
def generate_sent2(model, num_words, text_seed,random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    content.extend(text_seed)
    for token in model.generate(num_words,text_seed=text_seed, random_seed=random_seed):
        # print(token)
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)
#%%
maxN=[1,2,3,5]
for i in maxN:
    # print(odysseytext[0])
    print('-------This is from ody maxN is: ',i)
    od_corpus = list(word_tokenize(odyssey))
    paddedLine = [list(pad_both_ends(od_corpus, n=i))]
    mynltk_ngramtraining_data, mynltk_padded_sentences =nltk.lm.preprocessing.padded_everygram_pipeline(i, paddedLine)
    # Create the maximum-likelihood n-gram estimate
    my_nltk_ngrammodel = nltk.lm.MLE(i)
    my_nltk_ngrammodel.fit(mynltk_ngramtraining_data, mynltk_padded_sentences)
    # print(my_nltk_ngrammodel.counts['this'])
    gnerated = generate_sent(my_nltk_ngrammodel,100, random_seed=3)
        # Print the result
    
    print(gnerated)

for i in maxN:
    # print(odysseytext[0])
    print('-------This is form robin maxN is: ',i)
    rb_corpus = list(word_tokenize(robin))
    paddedLine = [list(pad_both_ends(rb_corpus, n=i))]
    mynltk_ngramtraining_data, mynltk_padded_sentences =nltk.lm.preprocessing.padded_everygram_pipeline(i, paddedLine)
    # Create the maximum-likelihood n-gram estimate
    my_nltk_ngrammodel = nltk.lm.MLE(i)
    my_nltk_ngrammodel.fit(mynltk_ngramtraining_data, mynltk_padded_sentences)
    # print(my_nltk_ngrammodel.counts['this'])
    gnerated = generate_sent(my_nltk_ngrammodel,100, random_seed=3)
        # Print the result
    
    print(gnerated)
#%%
maxN=[2,3]
for i in maxN:
    # print(odysseytext[0])
    print('-------This is from ody maxN is: ',i)
    od_corpus = list(word_tokenize(odyssey))
    paddedLine = [list(pad_both_ends(od_corpus, n=i))]
    mynltk_ngramtraining_data, mynltk_padded_sentences =nltk.lm.preprocessing.padded_everygram_pipeline(i, paddedLine)
    # Create the maximum-likelihood n-gram estimate
    my_nltk_ngrammodel = nltk.lm.MLE(i)
    my_nltk_ngrammodel.fit(mynltk_ngramtraining_data, mynltk_padded_sentences)
    # print(my_nltk_ngrammodel.counts['this'])
    gnerated = generate_sent2(my_nltk_ngrammodel,100,['the','moon'], random_seed=3)
        # Print the result
    
    print(gnerated)

for i in maxN:
    # print(odysseytext[0])
    print('-------This is form robin maxN is: ',i)
    rb_corpus = list(word_tokenize(robin))
    paddedLine = [list(pad_both_ends(rb_corpus, n=i))]
    mynltk_ngramtraining_data, mynltk_padded_sentences =nltk.lm.preprocessing.padded_everygram_pipeline(i, paddedLine)
    # Create the maximum-likelihood n-gram estimate
    my_nltk_ngrammodel = nltk.lm.MLE(i)
    my_nltk_ngrammodel.fit(mynltk_ngramtraining_data, mynltk_padded_sentences)
    # print(my_nltk_ngrammodel.counts['this'])
    gnerated = generate_sent2(my_nltk_ngrammodel,100,['the','moon'], random_seed=3)
        # Print the result
    
    print(gnerated)













