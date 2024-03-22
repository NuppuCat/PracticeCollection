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

% load true filter coefficients Wt_s1,Wt_s2 for NLMS performance analysis
% in Q9
load('Wt.mat');

% % % Q1 and Q2
mu=2.5e-10;
% use LMS from LMS.m
[e, w, Wt] = lms(s2, v, mu, M);
N = length(s1);
t=sum((e(N/4:N)-c(N/4:N)).^2);
b = 3*N/4+1;
mse1 = t/b;
% % % Q4 and Q5
mu=0.04;
a=0.01;
% implement NLMS based on the template of LMS.m
[e, w, Wt] = nlms(s2, v, mu, M,a);
t=sum((e(N/4:N)-c(N/4:N)).^2);
b = 3*N/4+1;
mse2 = t/b;
%%
[e, w, Wt] = lms(s2, v, 2.6667e-10, M);
t=sum((e(N/4:N)-c(N/4:N)).^2);
b = 3*N/4+1;
mse3 = t/b;
[e, w, Wt] = lms(s2, v, 2.4667e-10, M);
t=sum((e(N/4:N)-c(N/4:N)).^2);
b = 3*N/4+1;
mse4 = t/b;
%%
[e, w, Wt] = nlms(s1, v, 0.039333, M,a);
t=sum((e(N/4:N)-c(N/4:N)).^2);
b = 3*N/4+1;
mse5 = t/b;
[e, w, Wt] = nlms(s1, v, 0.029223, M,a);
t=sum((e(N/4:N)-c(N/4:N)).^2);
b = 3*N/4+1;
mse6 = t/b;
%%
[e, w, Wt] = nlms(s2, v, 0.046667, M,a);
t=sum((e(N/4:N)-c(N/4:N)).^2);
b = 3*N/4+1;
mse6 = t/b;
firgure;
plot(Wt(64));
%%
u = [zeros(M-1, 1); v];
uu = u(1+M-1:-1:1);
% % % HINT: for Q6 and Q7 use "linspace(0.01e-8,0.12e-8,16)";