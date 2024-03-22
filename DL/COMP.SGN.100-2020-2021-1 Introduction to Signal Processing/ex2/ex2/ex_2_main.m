%task 5
disp('task 5')
 delta = [zeros(1,7),1,zeros(1,7)];
 n = -7:7;
 figure(1);
 k=stem(n, delta);
 set(k, 'BaseValue', -0.2);
 u = [zeros(1,7),ones(1,8)];
 figure(2);
 stem(n,u) ;
 r = [zeros(1,7),0:7];
 figure(3);
 stem(n,r);
 %task 6
 disp('task 6')
 A = reshape(1:100,10,10);
 A = A';
 A3 = power(A,3);
 A3
 B = rand(10,10)
 C = inv(B);
 B*C
 %task 7 
 disp('task 7')
 audiosig=audioread('seiska.wav');
 figure(4);
 spectrogram(audiosig);
 %soundsc(audiosig);
 h = fir1(30, 0.3, 'high');
 y = filter(h,1,audiosig);
 figure(5);
 spectrogram(y);
 %soundsc(y);
  %task 8
  disp('task 8')
 n = 1:1000;
 x1=sin(2*pi*1000*n/8192);
 n = 1:2000;
 x2=sin(2*pi*2000*n/8192);
 n = 1:3000;
 x3=sin(2*pi*3000*n/8192);
 %soundsc(x1);
 %soundsc(x2);
 %soundsc(x3);
 n = 1:6000;
 x4=sin(2*pi*6000*n/8192);
 n = 1:7000;
 x5=sin(2*pi*7000*n/8192);
 n = 1:8000;
 x6=sin(2*pi*8000*n/8192);
 
 %soundsc(x4);
 %soundsc(x5);
 %soundsc(x6);
 disp('aliasing occur')
 %task9
 disp('task 9')
 load('seiska.mat')
 %soundsc(x,16384);
 y=x(1:2:length(x)); 
 %soundsc(y,8192);
 y =decimate(y,2);
 %soundsc(y,4096);
 %like a way to compress sgnal
 %task10
 disp('task 10')
 load gong
 coe =[ -0.2427 -0.2001 0.7794 -0.2001 -0.2427];
 z = conv(y,coe);
 %soundsc(y);
 %soundsc(z);
 hall = audioread('hall.wav');
 %soundsc(hall);
 soundsc(conv(hall,audiosig));
 