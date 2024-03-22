function [N,D] = design_lowpass(fp, fs, Rp, Rs, Fs)
wp = fp/Fs*2*pi;
ws = fs/Fs*2*pi;
omegap = 1;
c = 1/tan(wp/2);
omegas = c*tan(ws/2);
cp = (10^(Rp/10)-1)^0.5;
A = (10^(Rs/10))^0.5;

M = ceil(log10((A^2-1)/cp^2)/(2*log10(omegas)));

n = 1:M;
p = 1/(cp^(1/M))*exp(pi*i*(0.5+(2*n-1)/(2*M)));

pd = (1+p/c)./(1-p/c);

zd = -1*ones(M,1);
%disp(pd);

[N,D] = zp2tf(zd,pd,1);
k = sum(N)/sum(D);
%[N,D] = zp2tf(zd,pd,k);
N = N/k;


end