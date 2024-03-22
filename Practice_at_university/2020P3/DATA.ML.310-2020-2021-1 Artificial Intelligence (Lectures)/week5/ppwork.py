# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 18:42:32 2021

@author: onepiece

As preprocessing, you should remove all capitalization, special characters, and numbers from the text. In addition, remove the following common words (called ”stopwords”): [a, an, and, as, at, for, from, in, into, of, on, or, the, to]

After tokenization, remove also tokens with length <2. We perform this step because if e.g. you have in the text words like ”sister’s”, tokenization will result in meaningless tokens like ”s” (Note that this might lead to omitting words such as the personal pronoun ”I” which could be undesirable in practice, but for this exercise it is okay).

Using a word bi-gram language model, what is the probability of the phrase ’with her’?

Remove the leading zeroes from your answer, and consider the three most significant digits of the result as an integer (e.g. if your answer is 0.00010587, consider only 105 from it). Divide that integer with 59 and submit the remainder.

作为预处理，您应该从文本中删除所有大写字母，特殊字符和数字。另外，请删除以下常用词（称为“停用词”）：[a，an和as，at，for，from，in，in，of，on或on至]

标记化后，还要删除长度小于2的标记。我们执行此步骤是因为您在文字中有“ sister's”之类的词，则标记化将导致无意义的“ s”之类的标记（请注意，这可能会导致省略人称代词“ I”之类的词，这在实践中可能是不可取的，但对于本练习而言，没关系）。

使用两字语法语言模型，短语“和她在一起”的概率是多少？

从答案中删除前导零，然后将结果的三个最高有效数字视为一个整数（例如，如果您的答案是0.00010587，则仅考虑其中的105）。将该整数除以59，然后提交余数。
"""
import re
import nltk




data =[]
with open("The Story of An Hour - Kate Chopin.txt","r") as f:
    data = f.read()
    data = data.lower()
    data = re.findall('[a-z]+',data)
    # print(data)
    
# for word in words:
# words = []    
listf = ['a', 'an', 'and', 'as', 'at', 'for', 'from', 'in', 'into', 'of', 'on', 'or', 'the', 'to']
# def checkform(s):
#     if (s in listf)  or (len(s)<2):
#         return 1==0
S = filter(lambda x: (x not in listf) and (len(x)>=2),data)
# s = 'an'
# if s in  ['a', 'an', 'and', 'as', 'at', 'for', 'from', 'in', 'into', 'of', 'on', 'or', 'the', 'to'] or len(s)<2:
#     print(1)
words = list(S)
# for word in data:
#     if len(word)>2:
#         words.append(word)

bi = []
count = 0
for i in range(len(words)-2):
    bi.append([words[i],words[i+1],words[i+2]])
    if words[i]=='she' and words[i+1]=='did' and words[i+2]=='not':
        count =count+1
a = count/(len(bi)+2)
b = 375%127
# a = count/len(bi)
# b=376%59
print(b)
a = []
with open("The Story of An Hour - Kate Chopin.txt","r") as f:
    a = f.read()
    
nltk.download('punkt')    
result = nltk.word_tokenize(a)
print(result)



























               
               
               
               