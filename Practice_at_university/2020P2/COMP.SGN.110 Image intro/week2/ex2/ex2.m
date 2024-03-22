%task1
disp('task1')
I = imread('peppers.png');
A = [0:1:255];
%set length of per zone
step = 256/4;
%simple set zone by step,and returns values from X rounded to nearest multiple of step.
QA1 = quant(A, step);
%Create the quantization partition. To specify a partition, list the distinct endpoints of the different ranges of values.
partition = step:step:256-step;
%Specify the codebook values, as the value of every zone
codebook = step/2:step:256-step/2;
%Perform quantization on the sampled data. 
%indx means the elements belong to which zone;QA2 is the result of
%quantization, split A in zones by upper partition, set all value in same zone
%by codebook
[indx,QA2] = quantiz(A, partition, codebook);
disp(unique(QA1))
disp(unique(QA2))
[m n p] = size(I);
[index,QA] = quantiz(reshape(I ,[1,m*n*p ]), partition, codebook);
figure(1)
imshow(uint8(reshape(QA,[m,n,p])));

%%
%task2
clear all;
disp('task2')
L = imread('lena_face.png');
[m n]=size(L)
rL = reshape(L,[1 m*n]);
for level=[128,64,32,16,8,4]
   step = 256/level;
   [indx,QA] = quantiz(rL, step:step:256-step, step/2:step:256-step/2);
   qL = reshape(QA,[m n]);
   figure(level)
   imshow(uint8(qL))
end
nL = double(rL) + 10*randn(size(rL));
step1 = 256/16;
[indx,QA16] = quantiz(nL, step:step:256-step, step/2:step:256-step/2);
qL = reshape(QA16,[m n]);
figure(2)
imshow(uint8(qL))
%%
%task3
clear all;
disp('task3')
L = ones([400,400])*63;
L([(400-160)/2:(400-160)/2+160],[(400-160)/2:(400-160)/2+160])=127;
L = uint8(L);
R = ones([400,400])*223;
R([(400-160)/2:(400-160)/2+160],[(400-160)/2:(400-160)/2+160])=127;
R = uint8(R);
figure(3)
imshow(L)
figure(5)
imshow(R)
L1 = L;
L1([(400-160)/2:(400-160)/2+160],[(400-160)/2:(400-160)/2+160])=127+27;
figure(6)
imshow(L1)
disp('maybe plus 27?')