Alphabet = 1:6;
ns = 8; % number of symbols
%% Assume the probabilities are
ProbSymbols = [1/9 2/15 2/15 8/45 2/9 2/9]

%% Create the dictionary of a Huffman code using Matlab built in function
[ dict, avg_len ] = huffmandict( Alphabet, ProbSymbols );

CodeWords.a = dict{1,2}; CodeWords.b = dict{2,2}; CodeWords.c = dict{3,2};
CodeWords.d = dict{4,2}; CodeWords.e = dict{5,2}; CodeWords.f = dict{6,2};

CodeWords
%%
avgl=sum(ProbSymbols.*[3 3 3 3 2 2])