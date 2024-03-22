clear all;
S = "aacbdacbbbbadbaababadacbcadcbbbacaacaabb";
L = strlength(S);
a = count(S,'a')/L;
b= count(S,'b')/L;
c=  count(S,'c')/L;
d=  count(S,'d')/L;
H = (-a*log2(a)-b*log2(b)-c*log2(c)-d*log2(d))
LL=H*L
%%
% Original string:
S = 'aacbdacbbbbadbaababadacbcadcbbbacaacaabb'
% empirical counts
counts = [sum(S=='a') sum(S=='b') sum(S=='c') sum(S=='d')]
% number of symbols
n= length(S)
[length(S) sum(counts)]
% (empirical) probability of occurence of symbols in the string
p0 = counts/sum(counts)
% 2. Compute Total ideal length of the bitstream as sum of ideal codelengths
TotalIdealLength = 0;
for i = 1:length(S)
symbol_i = S(i);
ind_symbol_i = find( symbol_i == 'abcd')
ideal_codelength_i = -log2( p0( ind_symbol_i ) ); % -log2(p_i) is the ideal codelength
TotalIdealLength = TotalIdealLength + ideal_codelength_i;
end
TotalIdealLength
AverageIdealCodelength = TotalIdealLength/length(S) % bits per symbol
% Recompute the same, but reorganize the computations
TotalIdealLengthi1 = 0;
for c = 'abcd' % go over all symbols
indices = find( S == c ); % where symbol i appears in S?
n_i = length( indices) % how many times i appears in S
ind_symbol_c = find( c == 'abcd')
ideal_codelength_i = -log2( p0( ind_symbol_c ) );
TotalIdealLengthi1 = TotalIdealLengthi1 + n_i*ideal_codelength_i;
end
TotalIdealLengthi1
% Final recomputation, resulting in the formula of the entropy
n = length(S);
AverageIdealCodelength2 = 0;
for c = 'abcd' % go over all symbols
indices = find( S == c ); % where symbol i appears in S?
n_i = length( indices) % how many times i appears in S
ind_symbol_c = find( c == 'abcd')
ideal_codelength_i = -log2( p0( ind_symbol_c ) );
AverageIdealCodelength2 = AverageIdealCodelength2 + (n_i/n)*ideal_codelength_i;
end
AverageIdealCodelength2
%%
%% Encode the string S in a more "imaginative way", by symbol substitution
% First pick a string S
clear all;
a = 1, b = 2, c = 3
S = [a a c b a b b b b a b a a b a b a a c b c a c b b b a c a a ]
nS = length( S)
h0 = hist(S,1:3)
% (empirical) probability of occurence of symbols in the string
p0 = h0/sum(h0)
% entropy for the distribution p0
H0 = -sum( p0.*log2(p0) )
TotalIdealLength = nS * H0
% First replace in S the symbols 1 and 2 by symbol 6
% We end up with a string denoted S_3_6
S_3_6 = S;
S_3_6( S == 1) = 6;
S_3_6( S == 2) = 6;
% Evaluate the probabilities of 3 and 6
h_3_6 = hist(S_3_6,[3 6])
p_3_6 = h_3_6/sum(h_3_6)
% entropy for the distribution p_3_6
H_3_6 = -sum( p_3_6.*log2(p_3_6) )
% Evaluate the ideal number of bits for encoding S1
TotalForS_3_6 = length(S_3_6)*H_3_6
%% Now we go on to refine the information we got from S_3_6
% Clarify for each 6, was it a 1 or a 2 in the string S?
ind6 = find( S_3_6 == 6 );
S_1_2 = S(ind6);
% We now will encode the string S2
h_1_2 = hist(S_1_2,1:2);
p_1_2 = h_1_2/length(S_1_2);
% entropy for the distribution p6_12
H_1_2= -sum( p_1_2.*log2(p_1_2) );
% Evaluate the total bits for encoding S2
TotalForS_1_2 = length(S_1_2)*H_1_2;
[ TotalIdealLength TotalForS_3_6+TotalForS_1_2]
% Now divide by nS
[ TotalIdealLength TotalForS_3_6+TotalForS_1_2]/nS
% Compare to
[H0 H_3_6 H_1_2]

% What change is needed to get equality between
% [ TotalIdealLength TotalForS_3_6 TotalForS_1_2]/nS and [H0 H_3_6 H_1_2] ?