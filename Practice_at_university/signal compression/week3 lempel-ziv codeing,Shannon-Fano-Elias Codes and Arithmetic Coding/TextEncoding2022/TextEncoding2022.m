%% Compressing a text: Entropy, Arithmetic coding with memory, Lempel-Ziv as in gzip
%% Stage 0. Read a book from the file book1 (HARDY Madding Crowd, 1874)
% make sure you are in the folder TextEncoding (extracted from
% TextEncoding.zip)
clear all;
cd '../TextEncoding2022'
fid=fopen('book1.txt','rb');
txt=fread(fid);
fclose(fid);
len=length(txt);
dir_book = dir('book1.txt');
dir_book

%% Each entry in the array txt is one ASCII symbol (for letters, digits, punctuation signs)
% Q0: What is the size of the original book in bytes? Cross-check with the size on disk of the file book1.txt


%% Stage 1. Inspect the type of content of the book 
char(txt(1:1000))'  % Check the displayed text
alphabet = unique(txt) % find the used alphabet 
len_alphabet = length(alphabet)
char(alphabet);
empirical_freq = histc(txt,alphabet);
[  char(alphabet) repmat('  ',len_alphabet,1) num2str(empirical_freq) ...
    repmat('  ',len_alphabet,1) num2str(double(alphabet))  repmat('  ',len_alphabet,1) ...
    num2str(empirical_freq/sum(empirical_freq))]
[vs,is] = sort( empirical_freq, 'descend' );
char(alphabet(is(1:10)))
[char(alphabet(is)) repmat('  ',len_alphabet,1) num2str(empirical_freq(is)/sum(empirical_freq))]

%% Q1.1: The frequency of the letters is typical for English usage. (you could google "frequency of English letters")%% Q1.2: The letter e appears about once out of every twenty symbols
%% Q1.3: The least frequent symbol in this book is *, which in a programming text will appear very often
%% Q1.4: The capital letter Q is the capital letter the least often  appearing in this book 

%% Stage 2. Compute the entropy corresponding to symbol frequencies
empirical_probs = empirical_freq/sum( empirical_freq);
H = -sum( empirical_probs.* log2(empirical_probs))


% Transform all capital leters of txt into small letters
char(alphabet(32:56)) % This are the capital letters 
txt1 = txt;
for i1 = 32:56
    ind1 = find(txt1 == alphabet(i1));
    txt1(ind1) = char(double(txt1(ind1))+32); % transform the capital letter into small letter
end
char(txt1(1:1000))'
% Compute new probabilities for the restricted alphabet
alphabet1 = unique(txt1);
empirical_freq1 = histc(txt1,alphabet1);
empirical_probs1 = empirical_freq1/sum( empirical_freq1);
H1 = -sum( empirical_probs1.* log2(empirical_probs1))

[ dict, avg_len ] = huffmandict( 1:length(empirical_probs), empirical_probs );
avg_len
[ dict1, avg_len1 ] = huffmandict( 1:length(empirical_probs1), empirical_probs1 );
avg_len1

%% Q2.1:Using a prefix code for encoding the symbols (with no memory model 
%% included) one cannot encode the entire book with less than 4.5 bits per symbol 
%% Q2.2 If all capital letters of the text are transformed into small letters the 
%% new entropy is smaller than the old entropy
%% Q2.3 Encoding the book using a Huffman code designed on the empirical 
%% probability produces results very close to the corresponding entropy, 
%% within a difference of 1 percent


%% Stage 3. Evaluate the codelength for encoding by arithmetic coding 
%% using context models of order 0, 1, and 2
%% Use arithmetic coding with the Laplace adaptive probability model

%% Stage 3.0 Use 0-order model (encode by symbol probabilities, as seen so far)
size0=32; % encode first the length of the file using 32 bits
cnt=ones(256,1); % the counts of symbols encoded so far are updated in this counters
for i=1:len(1)
    x=txt(i);               % current symbol to encode
    p=cnt(x+1)/sum(cnt(:)); % probability of the symbol, as given by the current counters
    cl=-log2(p);            % ideal codelength, which is closely obtained by arithmetic coding
    size0=size0+cl;         % the current codelength of the encoded file
    cnt(x+1)=cnt(x+1)+1;    % update the count of the current symbol
end
size0_bytes=ceil(size0/8);  % size of the encoded file, in bytes
size0_bytes
[size0/len H]            % comparison of entropy to the bits/symbol of the encoded file

%% Q3.0: For a zero order memory model, the achieved size using arithmetic coding is better than the entropy, within 1 percent distance
%% Q3.1: For a zero order memory model, the achieved size using arithmetic coding is about 4.53 bits_per_symbol.
  

%% Stage 3.1 Use first-order model (encode by symbol probabilities conditional on the previous symbol, as seen so far)
size1=32; % length of the file
cnt=ones(256,256);
size1=size1+8; % the first byte is transmitted as it is
for i=4:len
    y=txt(i);                       % current symbol to be encoded
    x=txt(i-1);                     % the context of the current symbol
    p=cnt(x+1,y+1)/sum(cnt(x+1,:)); % probability of the symbol, as given by the current counters at the current context
    cl=-log2(p);
    size1=size1+cl;
    cnt(x+1,y+1)=cnt(x+1,y+1)+1;    % update the counts of the current symbol encountered at the context y 
end
size1_bytes=ceil(size1/8);
size1_bytes
[size1/len size0/len H]

%% Q3.1.1: The number of bits per symbol for this typical literature text is  about 3.7, if a first order memory model is used.
%% Q3.1.2: If one uses as a context txt(i-2) instead of txt(i-1) the average codelength is about 4.0
%% Q3.1.3: If one uses as a context txt(i-3) instead of txt(i-1) the average codelength is about 4.1
%% Q3.1.4: The fact that the average codelength per symbol using AC and model order 1 is better than 
%% the entropy of the model of order 0 is against the Shannon theorem presented in Topic 1

%% Stage 3.2 Use second-order model (encode by symbol probabilities conditional on the previous symbol, as seen so far)
size2=32; % length of the file
cnt=ones(256,256,256);
size2=size2+8+8; % the first two bytes are uncoded
for i=4:len
    x=txt(i-2);
    y=txt(i-1);
    z=txt(i);       % current symbol to be encoded
    p=cnt(x+1,y+1,z+1)/sum(cnt(x+1,y+1,:));
    cl=-log2(p);
    size2=size2+cl;
    cnt(x+1,y+1,z+1)=cnt(x+1,y+1,z+1)+1;
end
size2_bytes=ceil(size2/8);
size2_bytes
[size2/len size1/len size0/len H]

%% Q3.2.1: The number of bits per symbol for this typical literature text is  about 3.43 if a second order memory model is used.
%% Q3.2.2: If one uses as a context x=txt(i-3) instead of x=txt(i-2) the average codelength is about 3.81 (worse than for a simple first order model)
%% Q3.2.3: The worst performance of the context (y=txt(i-1),x=txt(i-3)) than the context txt(i-1) alone is due
%% to the fact that the former context is more diluted, i.e. it appears less often in the text, and hence the probability estimates are poorer


%% Stage 4. Encode the file book1.txt using Lempel-Ziv, as implemented in gzip of matlab
gzip('book1.txt')
S = dir('book1.txt.gz')
size_gzip = S.bytes*8 
[size_gzip/len size1/len size0/len H]

%% Split the file into N parts and encode them separately by gzip
for N = 1:4 % number of parts
    % set the segments' boundaries
    sizes_small = [0 floor((1:(N-1))*len/N) len];
    size_gzip_segm = [];
    for i = 2:(N+1)
        % Encode by gzip the current segment
        segment_to_encode = (sizes_small(i-1)+1):sizes_small(i);
        fid=fopen('book1_segm.txt','wb');
        fwrite(fid, txt(segment_to_encode));
        fclose(fid);
        gzip('book1_segm.txt')
        S = dir('book1_segm.txt.gz');
        size_gzip_segm(i-1) = S.bytes*8;
    end
    bitspersymb(N) = sum(size_gzip_segm)/len;
end
[bitspersymb size_gzip/len size1/len size0/len H]

    

%% Q4.1: The number of bits per symbol for a typical literature text is  about 3.26 if a dictionary based method is used.
%% Q4.2: If one splits the file into four equal parts and encodes each part separately, the number of bits per symbol becomes 3.30
%% Q4.3: The loss in performance when encoding four small parts instead of the entire text at once, is expected, because some of the useful regularities in the text,
%% as captured by the four implicit dictionaries, have to be learned again by each dictionary
%% Q4.4: The performance when gzip is used on (n+1) segments is always better than when gzip is used on n segments



