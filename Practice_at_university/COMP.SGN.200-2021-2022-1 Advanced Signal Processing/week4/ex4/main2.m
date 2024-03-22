
%%
close all;
clear all;
N = [250:250:2500];
NrOfTrials = [10:10:100];
for n = N
  [FPE_order,AIC_order,MDL_order] = pick_k(n,100);
end
%%
for t = NrOfTrials
  [FPE_order,AIC_order,MDL_order] = pick_k(2500,t);
end
%%
[FPE_order,AIC_order,MDL_order] = pick_k(2500,1000);
 h = hist(FPE_order, unique(FPE_order));
 h2 = hist(AIC_order, unique(AIC_order));
 h3 = hist(MDL_order, unique(MDL_order));
 %%
 [FPE_order,AIC_order,MDL_order] = pick_k(250,100);
 %%
 