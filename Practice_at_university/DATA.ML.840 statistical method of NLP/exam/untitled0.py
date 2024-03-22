# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 13:43:48 2021

@author: One
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:39:48 2021

@author: One
"""

temp_fullpath ='D://GRAM//MasterProgramme//Tampere//DATA.ML.840 statistical method of NLP//exam//grimms-fairy-tales-lowercase.txt'

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

import re
import nltk
import numpy
odysseytext = nltk.Text(nltk.word_tokenize(odyssey))

odysseyuniqueresults=numpy.unique(odysseytext,return_inverse=True)
odysseyvocabularies=odysseyuniqueresults[0]
odysseyindices_in_vocabularies=odysseyuniqueresults[1]



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
maxN=[5]
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
    gnerated = generate_sent(my_nltk_ngrammodel,50, random_seed=3)
    print(gnerated)
    gnerated = generate_sent(my_nltk_ngrammodel,50, random_seed=2)
        # Print the result
    
    print(gnerated)


#%%













