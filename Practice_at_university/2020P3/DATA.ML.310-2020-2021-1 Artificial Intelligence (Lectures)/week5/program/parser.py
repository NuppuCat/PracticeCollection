import nltk
import sys
import re




#规则仅来自于文本已有的例子。块中NP若有，则其中N不包括在内
TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP
S -> N V
S -> N V NP

S -> N VP
S -> NP V
S -> N V NP 
S -> NP V NP
S -> N VP NP 
S -> S Conj S
S -> V NP

NP -> Det N

NP -> NP P N
NP -> N NP


VP ->  Adv V | V Adv
VP -> V P
NP -> P N | P NP
NP -> NP P NP

NP -> Det N Adv
NP -> Det Adj N
NP -> Det AA N
NP -> Det AA NP
AA -> Adj Adj | Adj Adj Adj | Adj Adj Adj Adj | Adv Adj  


"""
# NP -> N NP
# conj N VP NP
# NP -> N P N
# NP -> Det AA N
# NP -> NP Det NP
# NP -> NP P NP

# PP -> P NP
# VP -> VP PP
# VP -> V Adv

# VP -> VP NP
# AA -> Adj Adj |Adj
grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    sentence = sentence.lower()
    nltk.download('punkt')
    words = nltk.word_tokenize(sentence)
    words = filter(lambda x: bool(re.search('[a-z]', x)) ,words)
    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # print(tree.height())
    # tree.height()==3 and
    chunks = tree.subtrees(lambda t: (t.height() == 3 and t.label()=='NP') or ( t.height()==2 and t.label()=='N') )
    # a = list(chunks)
    # print(len(a))
    return chunks


if __name__ == "__main__":
    main()
