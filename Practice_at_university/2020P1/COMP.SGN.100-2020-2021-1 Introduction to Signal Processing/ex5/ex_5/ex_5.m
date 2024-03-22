%%
%task4
disp('task4')
clear
N = 41;
n = -20:20;
fc = 0.3;
Fs = 0.5;
h = 2*fc*sinc(n*2*fc);
h(21) = 2*fc;
figure(1)
%freqz(h)

[H W]=freqz(h);
plot(Fs*W/(2*pi),20*log10(abs(H)))
title('amplitude response')
disp('a) The attenuation of the first (the leftmost) oscillatory peak inthe stopband is -20')
n1 = -30:30;
h1 = 2*fc*sinc(n1*2*fc);
h1(31) = 2*fc;
figure(2)
%freqz(h1)
[H1 W1]=freqz(h1);
plot(Fs*W1/(2*pi),20*log10(abs(H1)))
title('amplitude response')
disp('b) The first peak value is still -20,but it get a little bit eraly on x axis')
n2 = -50:50;
h2 = 2*fc*sinc(n2*2*fc);
h2(51) = 2*fc;
figure(3)
%freqz(h2)
[H2 W2]=freqz(h2);
plot(Fs*W2/(2*pi),20*log10(abs(H2)))
title('amplitude response')
disp('c) The first peak value is still -20,but it get a little bit eraly on x axis.Change N cannot change the stopband attenuation  ')

%%
%task5
disp('task5')
clear
fp = 0.5;
fs = 5/8;
n = -12:12;
Fs = 8;
a = fir1(12,1/16+0.5,'low');
figure(4)
impz(a)
title('impulse response')
figure(5)
freqz(a)
figure(6)
[H W]=freqz(a);
plot(Fs*W/(2*pi),20*log10(abs(H)))
%%
%task6
disp('task6')
clear
load handel;
a = fir1(50,1100/4096,'low');
b = fir1(50,1650/4096,'high');
c = fir1(50,[1750/4096,3250/4096]);
low  = (500+750)/8192;
bnd = 2750/4096;
%'DC-1' is stopband
d = fir1(50,[low,bnd],'DC-1');
figure(7)
[H W]=freqz(a);
plot(Fs*W/(2*pi),20*log10(abs(H)));
title('a')
figure(8)
[H W]=freqz(b);
plot(Fs*W/(2*pi),20*log10(abs(H)));
title('b')
figure(9)
[H W]=freqz(c);
plot(Fs*W/(2*pi),20*log10(abs(H)));
title('c')
figure(10)
[H W]=freqz(d);
plot(Fs*W/(2*pi),20*log10(abs(H)));
title('d')
ya = conv(a,y);
yb = conv(b,y);
yc = conv(c,y);
yd = conv(d,y);
sound(yd);
%%
%task7
disp('task7')
clear
[N, Wn] = buttord (0.2, 7/20, 0.3, 45);
[a b] = butter(N,Wn);
figure(11)
impz(a,b);
figure(12)
[H W]=freqz(a,b);
plot(40000*W/(2*pi),20*log10(abs(H)));
figure(13)
[z,p,K] = butter(N,Wn);
disp(z)
disp(p)
zplane(z,p)
%%
%task8 & task9
disp('task8 & task9, see the code in design_low_pass.m')
clear

%%
%task10
disp('task10')
clear

[N,D] = design_lowpass(9000, 12500, 0.4, 25, 32)
figure(14)
zplane(N,D)
figure(15)
[H W]=freqz(N,D);
plot(32*W/(2*pi),20*log10(abs(H)));
%freqz(N,D)