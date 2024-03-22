clear all;
close all;
% clean speech
[c,fs] = audioread('clear_speech.wav', 'native');
c = double(c);

% unpertubed noise
[v,fs] = audioread('noise_source.wav', 'native');
v = double(v);

% room 1
[s1,fs] = audioread('speech_and_noise_through_room_1.wav', 'native');
s1 = double(s1);

% room 2
[s2, fs] = audioread('speech_and_noise_through_room_2.wav', 'native');
s2 = double(s2);

% filter length
M = 200;

% load data matrix A
load('A.mat')
%%
d = s1;
w = inv(A'*A)*A'*d;
e = d-A*w;
N = length(s1);
mse = 1/N*sum((e-c).^2);
%%
d = s2;
w = inv(A'*A)*A'*d;
e = d-A*w;
N = length(s2);
mse2 = 1/N*sum((e-c).^2);
%%

N=length(s1);
d1 = s1(1:N/2);
d2 = s1(N/2+1:end);
Aset= pre(2,v,M)
A1 = Aset{1};
A2 = Aset{2};
w1 = inv(A1'*A1)*A1'*d1;
e1 = d1-A1*w1;
w2 = inv(A2'*A2)*A2'*d2;
e2 = d2-A2*w2;
e = [e1;e2];
mse3 = 1/N*sum((e-c).^2);
%%
N=length(s2);
d1 = s2(1:N/2);
d2 = s2(N/2+1:end);
Aset= pre(2,v,M)
A1 = Aset{1};
A2 = Aset{2};
w1 = inv(A1'*A1)*A1'*d1;
e1 = d1-A1*w1;
w2 = inv(A2'*A2)*A2'*d2;
e2 = d2-A2*w2;
e = [e1;e2];
mse4 = 1/N*sum((e-c).^2);
%%
N=length(s2);
es = cell(10);
Aset= pre(10,v,M);
for i= 1:10
   di = s2(1+(i-1)*N/10:i*N/10) ;
   Ai = Aset{i};
   wi = inv(Ai'*Ai)*Ai'*di;
   es{i}=di-Ai*wi;
end
e = [es{1};es{2};es{3};es{4};es{5};es{6};es{7};es{8};es{9};es{10}];
mse12 = 1/N*sum((e-c).^2);
%%
N=length(s1);
es = cell(5);
Aset= pre(5,v,M);
for i= 1:5
   di = s1(1+(i-1)*N/5:i*N/5) ;
   Ai = Aset{i};
   wi = inv(Ai'*Ai)*Ai'*di;
   es{i}=di-Ai*wi;
end
e = [es{1};es{2};es{3};es{4};es{5}];
mse11 = 1/N*sum((e-c).^2);