%% Huffman Coding and Canonical Huffman Example

%% 1. The source has 8 symbols: a,b, ..., h (or as numbers: 1,2,...,8)
Alphabet = 1:8;
ns = 8; % number of symbols
%% Assume the probabilities are
ProbSymbols = 2.^(-[2 5 5 3 2 5 5 2])

%% Create the dictionary of a Huffman code using Matlab built in function
[ dict, avg_len ] = huffmandict( Alphabet, ProbSymbols );

CodeWords.a = dict{1,2}; CodeWords.b = dict{2,2}; CodeWords.c = dict{3,2};
CodeWords.d = dict{4,2}; CodeWords.e = dict{5,2}; CodeWords.f = dict{6,2};
CodeWords.g = dict{7,2}; CodeWords.h = dict{8,2};  
CodeWords

%     a: [0 0]
%     b: [0 1 1 0 1]
%     c: [0 1 1 0 0]
%     d: [0 1 0]
%     e: [1 1]
%     f: [0 1 1 1 1]
%     g: [0 1 1 1 0]
%     h: [1 0]

%% Quiz L2.1
% We know that Huffman dictionary is not unique, there are many
% dictionaries that are equivalent.
% Repeat the calling of the matlab function huffmandict several times, with
% the same arguments
% Is the returned dictionary the same all the times? Is this a useful
% feature, or not?

%% Generate a long string with the given source probabilities

N = 1000000; % total number of symbols 10^6
% apply the inverse transform sampling method (a basic random generation method)
rands = rand( N, 1);
SymbString = zeros(N,1);
LowEnd = 0;
for ip = 1:8
    HighEnd = LowEnd + ProbSymbols(ip);
    SymbString( (rands > LowEnd) & ( rands <= HighEnd ) ) = ip; 
    LowEnd = HighEnd;
end

% Check that the symbols 1,...,8 appear in the string SymbString with the 
% empirical probabilities close to ProbSymbols (the intended probabilities)
ustring = unique( SymbString );
countsi = histc( SymbString(:), ustring )
% the empirical probabilities in the string are:
[countsi/N ProbSymbols(:)]

%% Quiz L2.2
% Repeat the generation of the N symbols several times. Is the array
% "countsi/N" keeping the same element values or not? 
%%  
% Repeat the generation process for N= 1000, N= 10000; N= 100000; N= 10^6.
% The differences between the vector "countsi/N" and "ProbSymbols" are
% decreasing as N increases, or it is the opposite case?

%% Encode the string using the Huffman dictionary

tic
hcode = huffmanenco(SymbString(:),dict);
toc
% Elapsed time is 0.227818 seconds.
% The length of the bitstream is 
BitstreamLength = length(hcode)
% The average length achieved by the Huffman code is length(hcode)/N 
[length(hcode)/N avg_len]

%% Decode the string using the Huffman dictionary
tic
dsymbols = huffmandeco(hcode,dict);
toc
% Elapsed time is 5.166025 seconds.
%% Number of decoding errors
NumberDecodingErrorsH = sum( dsymbols(:) ~= SymbString ) 


%% Quiz L2.3
% The ideal average length expected for the Huffman code is avg_len. For
% this string of symbols the average lengths [length(hcode)/N avg_len]
% are different. Is the difference due to errors in encoding or to errors in
% decoding, or to something else?
% If the length N of the string increases, is the difference between 
% length(hcode)/N and avg_len increasing or decreasing?
% Notice the encoding times and decoding times. They are changing over
% each running, even if SymbString is the same. What can one say: is the
% encoding faster than decoding using the given matlab functions?


%% Consider the Canonical Huffman tree from pages 43-44 of lecture slides
% Building the cannonical Huffman code is explained at page 43,
% resulting in two elements:
% a) The array firstcode keeps the most important information for fast 
% decoding (enabled by a quick PARSING of the bitstream)
firstcode = [2 1 1 2 0];
% b) the Table at bottom of page 43 keeps information for recovering the
% symbols
TableCanHuff = [0 0 0 0
    'a' 'e' 'h' 0
    'd' 0 0 0
    0 0 0 0
    'b' 'c' 'f' 'g']

% Transform the symbols from alphanumeric to numbers in {0,...,7}
SymbolsTable = double(TableCanHuff )-96;
SymbolsTable(SymbolsTable < 0) = 0 %  
 
% SymbolsTable =
% 
%      0     0     0     0
%      1     5     8     0
%      4     0     0     0
%      0     0     0     0
%      2     3     6     7

%% Consider the same SymbString as used above with the matlab built-in huffmanenco
% But encode it with the tree shown at page 44

% prepare the dictionary according to "dict" syntax in huffmanenco
for symb = 1:8
    [row,col]= find(SymbolsTable == symb);
    dictCH{symb,1} = symb;
    crt_code = firstcode(row)+col-1;
    code_word  = bitget( crt_code, row:-1:1)
    dictCH{symb,2} = code_word;
    % pause
end

CodeWordsCH.a = dictCH{1,2}; CodeWordsCH.b = dictCH{2,2}; CodeWordsCH.c = dictCH{3,2};
CodeWordsCH.d = dictCH{4,2}; CodeWordsCH.e = dictCH{5,2}; CodeWordsCH.f = dictCH{6,2};
CodeWordsCH.g = dictCH{7,2}; CodeWordsCH.h = dictCH{8,2}; 
CodeWordsCH
%     a: [0 1]
%     b: [0 0 0 0 0]
%     c: [0 0 0 0 1]
%     d: [0 0 1]
%     e: [1 0]
%     f: [0 0 0 1 0]
%     g: [0 0 0 1 1]
%     h: [1 1]
    
%% L2.4
% The set of codewords in the generated dictionary dictCH is the same as in
% the initial dictionary dict?
% The codewords are the same as in Figure from page 44?

tic
hcodeCH = huffmanenco(SymbString(:),dictCH);
toc
% Elapsed time is 0.562487 seconds.
% The length of the bitstream is 
BitstreamLength = length(hcode)
% The average length achieved by the Huffman code is length(hcode)/N 
[length(hcodeCH)/N avg_len]

%% Decoding using the Canonical Huffman code
tic
BITSTREAM = hcodeCH;
Symbol1 = zeros(N,1);
iBit = 1; % next bit to be processed
isymbol = 0; % which symbol was decoded last
Depth = 1; 
CodeWord = 0;
while ( iBit <= length( BITSTREAM ) )
    CodeWord = CodeWord*2 + BITSTREAM(iBit);        
    %vv = [vv; iBit Depth CodeWord BITSTREAM(iBit)];
    iBit = iBit +1;
    if( CodeWord < firstcode(Depth) )
        Depth = Depth +1;
    else
        isymbol = isymbol + 1;
        DisplacementInTable = CodeWord-firstcode(Depth);
        Symbol1(isymbol) = TableCanHuff(Depth, DisplacementInTable+1);
        CodeWord = 0;
        Depth = 1;
    end
end
Symbol1 = Symbol1(1:isymbol);
SymbolsNum = double(Symbol1)-96;   
EllapsedTimeCH = toc
% EllapsedTimeCH =   0.079367
NumberDecodingErrors = sum( SymbolsNum(:) ~= SymbString)

%% Decode the string using the Huffman dictionary
tic
dsymbols = huffmandeco(hcodeCH,dictCH);
EllapsedTimeH = toc

[EllapsedTimeCH EllapsedTimeH]

%% L2.5
% The decoding time of the huffmandeco function is larger than the the one 
% of the proposed implementation of Cannonical Huffman?

% The Instruction at line 173 is executed less often than the group of
% instructions at lines 175-179

