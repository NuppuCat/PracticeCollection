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

lambda = 0.9998; % forgetting factor, use values in the range [0.998,1]

% Recursive least squares from Matlab's DSP toolbox
hRLS = dsp.RLSFilter(...
    'Length',M,...
    'Method','Conventional RLS',...
    'ForgettingFactor',lambda);

[y,e] = step( hRLS,v,s1 );
release(hRLS)
%%
N = length(c);
[y,e] = step( hRLS,v,s1 );
release(hRLS)
mse1 = 1/(3*N/4+1)*sum((e(N/4:N)-c(N/4:N)).^2);


[y,e] = step( hRLS,v,s2 );
release(hRLS)
mse2 = 1/(3*N/4+1)*sum((e(N/4:N)-c(N/4:N)).^2);

%%
% filter length
M = 200;

lambda = 0.99971; % forgetting factor, use values in the range [0.998,1]

% Recursive least squares from Matlab's DSP toolbox
hRLS = dsp.RLSFilter(...
    'Length',M,...
    'Method','Conventional RLS',...
    'ForgettingFactor',lambda);
[y,e] = step( hRLS,v,s2 );
release(hRLS)
mse3 = 1/(3*N/4+1)*sum((e(N/4:N)-c(N/4:N)).^2);
%%
% filter length
M = 200;

lambda = 0.99993; % forgetting factor, use values in the range [0.998,1]

% Recursive least squares from Matlab's DSP toolbox
hRLS = dsp.RLSFilter(...
    'Length',M,...
    'Method','Conventional RLS',...
    'ForgettingFactor',lambda);
[y,e] = step( hRLS,v,s2 );
release(hRLS)
mse4 = 1/(3*N/4+1)*sum((e(N/4:N)-c(N/4:N)).^2);




