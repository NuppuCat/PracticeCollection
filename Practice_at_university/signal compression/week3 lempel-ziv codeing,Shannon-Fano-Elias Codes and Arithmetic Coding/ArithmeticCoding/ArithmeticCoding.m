clear
%% Arithmetic coding principles
%% 
% Assume the probabilities of the symbols are known to both encoder and decoder
%
% Define the appearing symbols 
%% Pick a string to encode and build the cumulative probability table
% Choose as string to be compressed 'Arithmetic'
clear all;
s = 'BILL GATES'%
uniq = unique(s)
duniq = double(uniq)
ds = double(s)
hh = hist(ds,duniq)
%%
% Keep probability values with "prec" bits precision; Stop if any of them has
% probability 0
format long
prec =  52 % 16
probs = hh/sum(hh);
probst = floor(probs*2^prec)/2^prec
sp = sum(probst);
if (any(probs) <=0 )
    display('Error: all probabilities must be strictly poistive')
    STOP1
end
%%
% Construct the cumulative probability table (Low and High values for each
% symbol)
lows = 0; highs = probst(1);
for i = 2:length(duniq)
    lows(i) = lows(i-1)+ probst(i-1);
    highs(i) = highs(i-1)+ probst(i);
end
InitialTable = [probst' lows' highs']
%%
% we assume that both decoder and encoder have the table "InitialTable"
%% Perform encoding
%
LowEnc = 0; HighEnc = 1; Progress = [LowEnc HighEnc]; prob_message = 1;
for i = 1:length(ds)
    current_symbol = ds(i);
    j = find( duniq == current_symbol );
    los = lows(j); his = highs(j);
    current_interval = HighEnc-LowEnc;
    HighEnc = LowEnc + current_interval*his;
    LowEnc = LowEnc + current_interval*los;
    Progress = [Progress; LowEnc HighEnc];
    prob_message = prob_message * probs(j);
end
%%
% The processing done at the encoder is found in the table "Progress"
 Progress 
%% Choose as message any number between LowEnc and HighEnc
% Show binary and integer values for the variables
intLowEnc = floor(LowEnc*2^40)
binLowEnc = bitget( intLowEnc, 40:-1:1)
intHighEnc = floor(HighEnc*2^40)
binHighEnc = bitget( intHighEnc, 40:-1:1)
[binHighEnc' binLowEnc' (1:40)']
message = (LowEnc+HighEnc)/2

ind = find(binHighEnc~= binLowEnc,1)
ind1 = find( binLowEnc(ind:end) == 0,1);
ind2 = ind+ind1-1;
binLowEnc(ind2) = 1;
%%
% This is the chosen bitstream
binLowEnc(1:ind2)
%%
% This is the representation of the bitstream as a subunitary number
format long
message = binLowEnc(1:ind2)*(2.^(-(1:ind2)'))
%%
% The probability associated with the message is prob_message and the ideal
% codelength is - log2(prob_message) bits
ideal_codelength = - log2(prob_message)
 
%% Perform decoding
MessageDecoded = [];
crt_message = message;
for i = 1:length(ds)
    for j = 1:length(duniq)
        los = lows(j); his = highs(j);
        if( (los <= crt_message) && (his > crt_message) )
            break
        end
    end
    current_symb_interval = his-los;
    crt_message = (crt_message - los)/current_symb_interval;
    MessageDecoded = [MessageDecoded uniq(j)]
end
MessageDecoded


