% SNG-20016 Advanced Signal Processing
% Exercise 2 Optimal Wiener filters
%

clear
close all

% load the audio sample that is treated as the input signal
[d, fs] = audioread('Tamara_Laurel_-_Sweet_extract.wav', 'native');
d = double(d);

% simulate the channel output with infinite signal-to-noise ratio
[u, w_true] = simulate_channel(d, Inf);

% extract a segment from both signals
s_start = 8;
d = d(s_start*fs+1:(s_start+1)*fs);
u = u(s_start*fs+1:(s_start+1)*fs);

% Use xcorr() to obtain the biased and unbiased autocorrelation estimates
% of u. Use maximum lag value 300.
%
%
%
%
%
%
[r,lags] = xcorr(u,300,'unbiased');
% Write your Wiener filter implementation here using d and u. 
% Use unbiased estimate for cross-correlation vector p (xcorr(d,u)) 
% and matrix R, both with maximum lag value 5. Use toeplitz() for
% reordering the autocorrelation estimate into the matrix R.
% Compare the obtained w with the channel parameters w_true.
%
%
%
%
%
%
%
uT=u';

[p,lags] =xcorr(d,u,5,'unbiased');
p = p(6:end);
r = xcorr(u,u',5,'unbiased');
r =r(6:end);
R = toeplitz(r) ;
RI=inv(R);
w=RI*p;
fu = filter(w,1,u);
mse=1/length(d)*sum((d-fu).^2)


















