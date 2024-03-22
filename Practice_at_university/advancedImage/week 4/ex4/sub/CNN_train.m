clc
%clear all 
close all
clear all


%% training data
load reference_image_patch %%% import clean refrence images
load noisy_image_patch     %%% import noisy images

XTrain = zeros(50,50,1,2000);
XTrain(:,:,1,:) = images_noisy_patch;

YTrain = zeros(50,50,1,2000);
YTrain(:,:,1,:) = images_gt_patch;

idx = randperm(size(XTrain,4),100);

XValidation= XTrain(:,:,:,idx);
YValidation=  YTrain(:,:,:,idx);



%% please write your code for network structure here
CNN=[
    imageInputLayer([50 50 1])
    convolution2dLayer(3,64,'Padding',[1 1])
    reluLayer
    convolution2dLayer(3,64,'Padding',[1 1])
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,64,'Padding',[1 1])
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,1,'Padding',[1 1])
    regressionLayer
    ];

%% training parameters 
options = trainingOptions('adam', ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4, ...
    'Shuffle' , 'every-epoch',...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
     'MiniBatchSize',256, ...
    'Plots','training-progress');
%% train network , enter your designed CNN here 

net = trainNetwork(XTrain,YTrain,CNN,options);
