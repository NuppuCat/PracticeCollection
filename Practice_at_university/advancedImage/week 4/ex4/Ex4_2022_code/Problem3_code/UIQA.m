clear all;
clc
%% download the model and load it
 load('Alexnet27_26(ReLu)_25(fc10)_24(ReLu)_23(fc10).mat')


%% use UIQA metric for first 200 images of NRTID and calculate images scores 
% load NRTID.txt,  this file is containing MOS values of NRTID dataset.
load mos_NRTID.txt
len = size(mos_NRTID,1)-300; %total number of images are 500 and we need first 200 images 
%% 
UIQA_score = zeros(1,len);  %% initialize a score vector with zero. ( this vector should be filled with image quality score) 
%%% your task %%%% 
% read images
% use the model to calculate the image score(use UIQA_f function)
% complete function in spearmanCorr.m file so you can calulate SROCC and report it.

%%% start your code here 
dir = 'D:\GRAM\MasterProgramme\Tampere\advancedImage\week 4\ex4\Ex4_2022_code\Problem3_code\NRTID\';
for i =1:len
    p = [dir,int2str(i),'.jpg'];
    image=imread(p);
    UIQA_score(i)=UIQA_f(image, UIQAAlexopt_srocc);
end

%%% end your code here 

%%
% save('UIQA_score','UIQA_score'); %% uncomment if you want to save yout final score values
 fprintf('\n SROCC=%.2f \n',spearmanCorr(UIQA_score,mos_NRTID(1:200)));
