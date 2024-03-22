%% Borrows Wheeler coding principles
% See the lecture slides at the begining of Lecture for Topic 6

%% Stage I: The Burrows-Wheeler transformation ("encoding" algorithm)
clear all;
close all;
% Pick a text to compress
input_str ='illustration'
input_str ='corresponding'
% input_str = 'Check the QuizLect6 Description from EliasCode.html (in the archived files of Quiz Lecture6), where you can find also the published results of running the code. Execute the matlab code EliasCode.m step by step and stop at the questions of the quiz to extract your answers. For some questions you have to change slightly the matlab code. Answer which of the following statements are true (if any).'
inp = double(input_str(:)); % transform from alphanumeric to integers
N = length(inp);

%% Compute the Burrows-Wheeler "direct" transform

T=gallery('circul',inp); % This performs the circular shifts of inp
T=flipud(T); % The matrix P is reversed upside-down
[Ts,~, indP] = unique([T(:,(N-1):-1:1) T(:,N)],'row');
char([Ts(indP,:) T])
% treat the rare cases when some rows of the matrix T are identical
if( length( unique(indP)) == 1 ) 
    hh = N;
else
    hh = hist( indP, unique(indP));
end
PP = [];
for i = 1: length( unique(indP) )
    PP = [PP; repmat( Ts(i,:),hh(i),1)];
end
Ts = PP;
char(PP)
% Show the initial circulant matrix (T) and the sorted matrix PP, as in the
% lecture slides (slide 126 Topic 6)
CC=[];
for i = 1:N
CC = [CC; char(T(i,1:(N-1))), '    ',  char(T(i,N)),  '    ',char(PP(i,(N-1):-1:1)), '    ' char(PP(i,N)) ];
end
CC
%%
% The message to send is: first the column Pd=PP(:,N) and then the index p of
% the first row of the matrix (first symbol in the text)
Pd = PP(:,N); p = indP(1);
char(Pd(:))
p

%% Stage II The Inverse Burrows-Wheeler transformation ("decoding" algorithm)
%%
% Inspect the message for existing symbols, and initialize the vector M for
% each symbol
Ms=1;
for ii=1:max(Pd)
    K(ii) = sum(Pd==ii);
    if(K(ii)~=0), M(ii)=Ms; Ms=Ms+K(ii); char(ii); end
end
%%
% Initialize the array L
for i=1:N
    s=Pd(i);
    L(i)=M(s);
    M(s)=M(s)+1;
end
%%
% start from position p and continue decoding
Dec = [];
i=p;
for k=1:N
    Dec(k)=Pd(i);
    i=L(i);
end
char(Dec)
 
%% For the input string 'illustration', check the structure of the CC matrix (made of columns of T and PP, arranged to correspond to the explanations in the lecture) 
%% QQ1.1 The value p = 6 signifies that in the sorted matrix PP the row with index 6 corresponds to the 
%% original first row of T (in the sense that CC(6,20+(1:16)) is identical to CC(1,1:16))
%% QQ1.2 The value p = 6 signifies that in the original matrix T the row with index 6 corresponds to the original message
%% QQ1.3 The matlab function unique(T,'row') finds the unique rows in T and arranges them one on top of each other, sorted alphabetically from left to right
%% QQ1.4 If the input string has repeated symbols, the message Pd will contain only one occurence of each symbol
%% QQ1.5 The string to be transmitted to the decoder, Pd, has a lower entropy than the initial input_str
%% QQ1.6 The decoder needs to know the probability distribution of the symbols in advance, so this probability distribution needs to be transmitted as side information


%% Stage III Move to front (MTF) applied to the output of BWT encoder
% The MTF coding is explained in the lecture of Topic 6, at the slides 133-138

% Run the BWT encoder for the string
% input_str = 'Check the QuizLect6 Description from EliasCode.html (in the archived files of Quiz Lecture6), where you can find also the published results of running the code. Execute the matlab code EliasCode.m step by step and stop at the questions of the quiz to extract your answers. For some questions you have to change slightly the matlab code. Answer which of the following statements are true (if any).'
% the encoder output is Pd, which gives the following: EncStr = char(Pd')

EncStr = 'wAEFDccrfsaEsmfqcQapqmqtyt(tatQtrtsot(sEftftcebaawsaooyoatychstLtii,. Cmh   ) nhooexlloeeuulrnntnrnvnssbbttcty  looohatttkuhhr    eeeeexscc.. .. cd         rsn s  6mpp ss  drrcioir     he  atea iteeeeeeeeeeeifnlnvcaasgnpooozzzn iisieo liaaytaa e elsyds gt dggi ssnffffddrdddlnnnwmm   puuuu  t uuueuceie so ua etlttotCCc utthoww  .  rhohhhhhhhho lle u6mlssioaeeiieriit  rbieenelreehhieeetooo  )L  '

Pd = double( EncStr );
alphabet = unique(Pd(:));
char(alphabet)
list_symb = alphabet(:);
for k = 1:length(Pd)
    crt_symb = Pd(k);
    Index_crt = find( list_symb == crt_symb );
    list_symb( list_symb == crt_symb ) = [];
    list_symb = [crt_symb; list_symb];
    Index_crti(k) = Index_crt;
end
hh1 = hist( Index_crti, unique(Index_crti))

figure(1), plot( hh1 ,'o-b' )
xlabel( 'i, index in MTF list' )
ylabel( 'frequency of index i')

pp = hh1/sum(hh1);
H = - sum( pp.*log2( pp ))
    
%% QQ2.1 The initial list of the MTF algorithm is ' (),.6ACDEFLQabcdefghiklmnopqrstuvwxyz'
%% QQ2.2 The entropy of the index Index_crti is identical to the entropy of the sequence Pd
%% QQ2.3 For longer blocks of text the obtained entropy of Index_crti is better than the entropy for shorter blocks (see the Quiz on Bzip2 for results over long texts)
%% QQ2.4 In the sequence Pd (and corresponding EncStr) there are many repetitions of the same symbol, which can be encoded very efficiently by run length coding (see lecture of Topic 6, slides 139).
%% QQ2.5 The probability distribution of the output of MTF shown in Figure 1 is similar to the exponential distribution of integers for which Golomb Rice coding is optimal



