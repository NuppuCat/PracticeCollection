function [metr] = UIQA_f(im, net)
% UIQA_f - computes metric score for one input image
%
%   [metr] = UIQA_f(im, net)
%
%   im - input image;
%   net - network model;
%   metr - metric value
k=0;
    for ii = 1:56:size(im,1)-227
        for jj =1:56:size(im,2)-227
            k =k+1;
            patch(:,:,:,k)= im(ii:ii+226,jj:jj+226,:);  
            YPred(k) = predict(net,patch(:,:,:,k));
        end 
     end
    metr = mean(YPred);
end

