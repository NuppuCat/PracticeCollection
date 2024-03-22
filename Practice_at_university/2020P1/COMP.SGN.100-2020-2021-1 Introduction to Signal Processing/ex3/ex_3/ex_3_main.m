%task4
disp('task4')
n = -10:10;
u = [zeros(1,10),ones(1,11)];
%y = x(n)/1-1.1y(n-1) a=1 b=[1 -1.1] change +- of y's para 
y = filter(1,[1 -1.1],u);
figure(1);
stem(n,y);
%task5
disp('task5')
delta=[1,zeros(1,127)];
n = 0:127;
y1 = filter(1,[1 -0.75],delta);
figure(2);
stem(n,y1);
y2 = filter([0.2 -0.5 0.8],[1 -0.6 0.3],delta);
figure(3);
stem(n,y2);
y3 = filter([1 0.5 1.25],[1 0.8 -0.8],delta);
figure(4);
stem(n,y3);
disp('as the stem (c) is not stable')
%task6
disp('task6')
b = 1;
a = [1 -0.75];
[h,t] = impz(b,a);
figure(5);
plot(t,h);
b = [0.2 -0.5 0.8];
a = [1 -0.6 0.3];
[h,t] = impz(b,a);
figure(6);
plot(t,h);
b = [1 0.5 1.25];
a = [1 0.8 -0.8];
[h,t] = impz(b,a);
figure(7);
plot(t,h);
%task7
disp('task7')
n = 0:2000;
x1=sin(2*pi*2000*n/16000);
y = abs(fft(x1));
figure(8);
plot(n,y);
%task8
disp('task8')
x = [1 2 3 4]';
dft = dft(x)
%task9
clear
disp('task9')
 x = rand(1024,1);
 tic(); 
 X=dft(x);
 elapsed_time=toc()
 tic(); 
 X=fft(x);
 elapsed_time=toc()
 tic();
for i=1:100
    
   X=dft(x);
     
end
 elapsed_time=toc()
  tic();
for i=1:100
    
   X=fft(x);
     
end
 elapsed_time=toc()
  N = [32,64,128,256,512,1024];
  elapsed_time_dft = zeros(length(N));
  for i = 1:length(N)
      tic();
      for j=1:100
    
         X=dft(x(1:N(i)));
     
       end
      elapsed_time_dft(i)=toc();
  end
  elapsed_time_fft = zeros(length(N));
  for i = 1:length(N)
      tic();
      for j=1:100
    
         X=fft(x(1:N(i)));
     
       end
      elapsed_time_fft(i)=toc();
  end
  figure(9)
  
  plot(N,elapsed_time_dft,'r')
  hold on;
  plot(N,elapsed_time_fft,'b')
  hold off;
  %%
  %task10
  disp('task10')
  clear
  load('Ex3_Task10.mat');
  fx = fft(x);
  fy = fft(y);
  %fx fy
  %Ϊ��Ƶ����Ϊ�Ա�����������źŵĸ���Ƶ�ʳɷֵķ�ֵ��Ϊ�������������Ƶ�ʺ�����Ϊ��ֵ�ס���fy./fy���Եõ���ֵϵ����������fft���Եõ���Ӧ��ʱƵ�ź�
  fh = fy./fx;
  h = ifft(fh);
  figure(10)
  stem(1:10, h(1:10))
  
  %c =conv( x,h);
  %c(1:10)-y(1:10) ���Է���c��y�ڱ�������һ�µģ�֤�����������󣬵������ʱ��ȫΪ0���к�С�Ĳ������Ժ�ѧϰ������
  
  