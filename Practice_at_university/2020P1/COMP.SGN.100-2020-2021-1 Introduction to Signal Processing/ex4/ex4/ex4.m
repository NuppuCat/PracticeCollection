%%
%task 6 
disp('task6')
n = 0:70;
x = sin(0.05*2*pi*n);
y =  filter([0.0349 0.4302 -0.5698 0.4302 0.0349],[1],x);
y1 = 0.305*sin(0.05*2*pi*n-0.6283);
figure(1);
hold on;
plot(n,y,'rx');
plot(n,y1,'b');
plot(n,x,'g');
hold off;
%%
%task 7 
disp('task7')
clear
w = 2*pi*5000/20000;
%y2 = filter([0.0675 0.1349 0.0675],[1 -1.143 0.4128],x);
Hz = (0.0675*exp(2*w*1i)+0.1349*exp(w*1i)+0.0675)/(exp(2*w*1i)-1.143*exp(w*1i)+0.4128);
attenuation = 10*log(abs(Hz))
%%
%task 8
disp('task8')
clear
a = [0.0122 0.0226 0.0298 0.0204 0.0099];
b = [1 -0.9170 0.0540 -0.2410 0.1990];
figure(2)
zplane(a,b)
figure(3)
freqz(a,b)
figure(4)
zplane(b,a)
figure(5)
freqz(b,a)
load handel;
f = filter(a,b,y);
f = filter(b,a,f);
figure(6)
%x = 0:size(y)-1;
%plot(x,abs(fft(y)),'r')
spectrogram(y)
figure(7)
spectrogram(f)
%plot(x,abs(fft(f)),'g')
disp('They are similar.')
%%
%Task 9
disp('task 9')
clear
t=0:1/8192:4;
y=chirp(t,0,1,1000);
figure(8)
spectrogram(y)
%soundsc(y);
%pause(4);
y2 = filter([0.0675 0.1349 0.0675],[1 -1.143 0.4128],y);
%soundsc(y2);
figure(9)
spectrogram(y2)
%%
%Task 10
disp('task 10')
clear

load number.mat;
%sound(secret);s = spectrogram(x,window,noverlap,nfft)
figure(10)
spectrogram(secret)
a = [0.1702 0.1880 0.2080 0.2297];
b = [0.2952 0.3262 0.3606];
n = 0 : 7768;
x1 = sin(a(1)*pi*n)+sin(b(1)*pi*n);
x2 = sin(a(1)*pi*n)+sin(b(2)*pi*n);
x3 = sin(a(1)*pi*n)+sin(b(3)*pi*n);
x4 = sin(a(2)*pi*n)+sin(b(1)*pi*n);
x5 = sin(a(2)*pi*n)+sin(b(2)*pi*n);
x6 = sin(a(2)*pi*n)+sin(b(3)*pi*n);
x7 = sin(a(3)*pi*n)+sin(b(1)*pi*n);
x8 = sin(a(3)*pi*n)+sin(b(2)*pi*n);
x9 = sin(a(3)*pi*n)+sin(b(3)*pi*n);
x10 = sin(a(4)*pi*n)+sin(b(1)*pi*n);
x0 = sin(a(4)*pi*n)+sin(b(2)*pi*n);
x11 = sin(a(4)*pi*n)+sin(b(3)*pi*n);
%sound(x1)
figure(11)
spectrogram(x0)
figure(12)
spectrogram(x1)
figure(13)
spectrogram(x2)
figure(14)
spectrogram(x3)
figure(15)
spectrogram(x4)
figure(16)
spectrogram(x5)
figure(17)
spectrogram(x6)
figure(18)
spectrogram(x7)
figure(19)
spectrogram(x8)
figure(20)
spectrogram(x9)
figure(21)
spectrogram(x10)
figure(22)
spectrogram(x11)
disp('after check with figure(10),the number is 36533840')
