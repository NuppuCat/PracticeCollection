clear

% Figure Transfer of power through linear systems

figure(1)

%% Choose an AR model with which to generate data

n= 6; % number of zeros of the AR polynomial
m= 2;
j=sqrt(-1);
r1= 0.9; om1=pi/3; % modulus and phase of the first two roots
r2 = 0.7; om2 = 3*pi/4; % modulus and phase of the next two roots
r3 = 0.99; om3 = pi/6;  % modulus and phase of the last two roots
rootsi =[r1*exp(j*om1) r1*exp(-j*om1) r2*exp(j*om2) r2*exp(-j*om2) ...
    r3*exp(j*om3) r3*exp(-j*om3)]; % all roots
A = poly(rootsi); % the AR polynomial having the roots given in rootsi
nA = length(A);

N1 = 2048; % number of points in the frequency range [0; 2*pi)
omega0 = 2*pi/N1; % the resolution in frequency 
% the set of frequency points omega = ii*omega0
om = 0:omega0:(2*pi);

%% Compute the true spectrum of the AR process
%% Evaluate P(omega) = 1/|A(exp(i*omega))|^2

H = zeros(length(om),1);
for ii = 1:length(om)
    H(ii) = exp(-j*om(ii)).^(0:(nA-1))*A';
    H(ii) = 0;
    omega = om(ii);
    for iAR = 1:nA
        H(ii) = H(ii) + A(iAR)*(exp(-j*omega))^(iAR-1);
    end
    P(ii) = 1 ./ abs(H(ii)).^2;
end

figure(2),clf
subplot(222)
plot(om(1:N1/2),log10(P(1:N1/2)))
grid
title('True power spectrum of AR process','Fontsize',14)
xlabel('frequency $\omega$','interpreter','Latex','Fontsize',12)
ylabel('$\log P(\omega)$','interpreter','Latex','Fontsize',12)

%% Questions:
% Q1: The highest power spectrum value P(omega) is at the angular frequency:
% a) omega = pi/6; b) omega = pi/ 3; c) omega = 3*pi/4

%% Generate one realization of the AR process

N = 200000;  % Number of data samples to be generated
y = rand(n,1);
e = randn(N,1);
for ii = (n+1):N
    y(ii) = -sum( y(ii-(1:n))' .* A(2:end)) +e(ii);
end
% Keep only the last (N1 = 1024) data samples (the rest are discarded to be
% sure that the transient part of the AR process ended)
y1 = y((1:N1) + end-N1); % this is the realization further processed for getting the spectrum estimate
% Compute the FFT of the N1 = 1024 data points
Y1 = fft(y1);
P1 = abs(Y1).^2/N1; % this is the periodogram estimate
om1 = om(1:(N1/2));

subplot(223)
plot(om1,log10(P1(1:N1/2)))
grid
title('Estimated spectrum from one realization','Fontsize',18)
xlabel('frequency $\omega$','interpreter','Latex','Fontsize',16)
ylabel('$\log \hat P(\omega)$','interpreter','Latex','Fontsize',16)

% Q2: The highest estimated power spectrum value P1(omega) is at 
% an angular frequency close to 
% a) omega = pi/6; b) omega = pi/ 3; c) omega = 3*pi/4

subplot(224)
plot(om1,log10(P1(1:N1/2)),'b')
hold on
plot(om(1:N1/2),log10(P(1:N1/2)),'r-','LineWidth',3)
grid
title('Estimated overlapped with true spectrum','Fontsize',18)
xlabel('frequency $\omega$','interpreter','Latex','Fontsize',16)
ylabel('$\log \hat P(\omega)$, $\log \hat P(\omega)$','interpreter','Latex','Fontsize',16)


subplot(221)
NN = 400;
plot(y1(1:NN))
grid
title('One realization of AR process','Fontsize',18)
xlabel('  time $t$','interpreter','Latex','Fontsize',16)
ylabel('$y_t$','interpreter','Latex','Fontsize',16)

%% Questions:
% Q3: The plot of y1 shows an underlying repetitive process, 
% with a period T of about 
% a) 12 samples; b) 30 samples;c) 3 samples
% Q4: Knowing the relationship omega = 2*pi/T, the period you
% notticed at Q3 corresponds to the angular frequency having the
% a) lowest power; b) highest power c) DC term omega = 0;

 
%% Generate multiple realizations of the AR proces, find the associated 
%% FFT,and average over all realizations at each frequency value

N1 = 2048;
Pmean = zeros(N1,1);
N_realiz = 200
for i_realiz = 1:N_realiz
    y=rand(n,1);
    N= 200000;
    e= randn(N,1);
    for ii = (n+1):N
        y(ii) = -sum( y(ii-(1:n))' .* A(2:end)) +e(ii);
    end
    y1 = y((1:N1) + end-N1);
    Y1 = fft(y1);
    P1 = abs(Y1).^2/N1;
    Pmean = Pmean + P1;
end
Pmean = Pmean/N_realiz;
om1 = om(1:N1/2);


figure,clf,
plot(om1,log10(P1(1:N1/2)),'-b'),hold on
plot(om1,log10(Pmean(1:N1/2)),'-r'),hold on
grid
title('Estimated spectrum over many realizations','Fontsize',18)
xlabel('frequency $\omega$','interpreter','Latex','Fontsize',16)
ylabel('$\log \hat P(\omega)$','interpreter','Latex','Fontsize',16)
plot(om(1:N1/2),log10(P(1:N1/2)),'-k')

%Q5: By generating independent realizations and averaging the 
% spectrum estimate obtained at a given omega, as we do in Pmean,
% one gets a method similar to
% a) Bartlett method b) Welch method c) Daniell method
%Q6: The spectrum estimate Pmean is very close to the ideal spectrum P,
% however, there are differences between the two. How one can ensure that
% P and Pmean become closer:
% a) Use a larger number of realizations N_realiz; b) Use a larger value of N1;
% c) Use a smaller value of N1; d) Use a smaller number of realizations N_realiz;

  