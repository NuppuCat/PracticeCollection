%% Elias codes of the first 2^n integers
% (Elias Codes are presented at slides 142 to 145 in the lecture slides of Topic 6)

%%
% Find the binary represenations of the fiirst 2^n integers (each row in B
% will be a binary representation of an integer
clear
n = 16% 10; % change this number, if needed, but not higher than 20
B = [0;1];
nB = 2;
for i = 2:n
    B = [ zeros(nB,1) B; ones(nB,1) B];
    nB= nB*2;
end
% B
%% Find the Elias codewords for all integers between 1 and (2^n-1)
Elias = zeros((size(B,1)-1),1);
for ii = 1:(size(B,1)-1)
    integerii = ii;
    % take the binary representation of integer "ii"
    integeri = B(ii+1,:);
    %%
    % Find the most significant (leading) one in the binary representation: 
    in1 = find(integeri==1,1,'first');
    integeri(1:in1)=[];
    % The bits in the vector "integeri" will be appended to the last part
    % of the Elias codeword. The first part of the codeword describes "N",
    % which says how many bits we have in the binary vector "integeri"
    N = length(integeri);
    % Find the binary representation of "N"
    binaryN = B(N+1,:);
    in1 = find(binaryN==1,1,'first');
    binaryN(1:(in1-1))=[];
    % Now the binary vector "binaryN" gets transmitted, by using the
    % repetition coding
    if (integerii >= 4)
        % always the Elias codewords starts with 1, for integerii >= 4
        eliascodeword = 1;
        for ij = 2:(length(binaryN)-1)
            % now repeat twice each bit from binaryN, until next to last position
            eliascodeword = [eliascodeword binaryN(ij) binaryN(ij)];
        end
        if( binaryN(end) == 0)
            % the last bit is encoded by 01 or 10, depending on the bit
            % being 0 or 1
            eliascodeword = [eliascodeword 0 1];
        else
            eliascodeword = [eliascodeword 1 0];
        end
        % finally append the bits from the binary vector "integeri"
        eliascodeword = [eliascodeword integeri];
    else
        switch integerii
            case 1
                eliascodeword = [ 0 0];
            case 2
                eliascodeword = [ 0 1 0] ;
            otherwise
                eliascodeword = [ 0 1 1] ;
        end
    end
    Elias(ii) =  length(eliascodeword);
    if( integerii < 42 )
        Pair_Integer_Codeword = [ integerii eliascodeword ]
    end
end

%% Build the table with the codelength of codewords for Elias code
% Elias
hh = hist(Elias,1:24);
[(1:24)' hh']

figure(1),plot(Elias,'-b','Linewidth',1)
xlabel('Integer i')
ylabel('Length of Elias codeword for integer i')

figure(2),loglog(2.^(-Elias),'-b','Linewidth',1)
xlabel('Integer i')
ylabel('????')


%% Q L6.1 The Elias codeword for i=39 is 1     0     0     1     0     0     0     1     1     1
%% Q L6.2 The codelength for i = 890 is 17
%% Q L6.3 The number of codewords with length 20 is 8192 
%% Q L6.4 The number of codewords with length 22 is 16384 
%% Q L6.5 The y axis in Figure 2 should be marked as 'ideal probability distribution of integers, for which Elias codes are optimal'

%% Stage II Find an approximative expression for the probability law of integers, as assumed by the Elias codes

% In Figure 2 the resolution is not great. One can use the alternative plots: 

figure(3),semilogx(2.^(-Elias),'-b','Linewidth',1)
xlabel('Integer i')
ylabel('P(i)')

figure(4),semilogy(2.^(-Elias),'-b','Linewidth',1)
xlabel('Integer i')
ylabel('P(i)')

figure(5),clf, loglog(2.^(-Elias),'-b','Linewidth',1)
xlabel('Integer i')
ylabel('P(i)')

%% The plot in loglog coordinates hints to a dependency P(i) = a*i^b, for some a and b
% Let's find a and b by least squares

n1 = length(Elias);
P = 2.^(-Elias);
x = log10(1:n1);
y = log10(P);
% the model is y = x*C1+C2
RegrMat = [ x(:) ones(size(x(:)))];
RegrMatb = [ x(:).^2 x(:) ones(size(x(:)))];
% the LS estimates of C = [C1; C2] are
C = RegrMat\y(:)
Cb = RegrMatb\y(:)
figure(5),clf, loglog(2.^(-Elias),'-b','Linewidth',1),hold on
loglog(1:n1, 10.^(Cb(1)*x.^2+Cb(2)*x+Cb(3)),'.r','Linewidth',1)
loglog(1:n1, 10.^(C(1)*x+C(2)),'.','Linewidth',1)
a=0.40282 ; b = -1.5378
loglog(1:n1, a*(1:n1).^b,'.g','Linewidth',21)
xlabel('Integer i')
ylabel('P(i)')

                
%% Q L6.6 A reasonable LS approximation from the loglog plot for n= 10 is P(i) = a*i^b, for   a=0.40282 and b = -1.5378
%% Q L6.7 The LS approximation from the loglog plot for n= 10 is P(i) = a*i^b, for   a=0.0108 and b = +1.5378
%% Q L6.8 When using a higher value of the parameter n (first line in the program) one sees 
%% that the approximation P(i) = a*i^b found at n= 10 holds very well also for largers ranges of i


%% Stage III Bonus point (Facultative)
%% Q L6.9 If you can obtain a better approximation of the loglog plot, 
% say for n=16, or better for  n = 20, write a short report in the essay space of this quiz,
% presenting your derivations, and the plots of your better approximation on top of the
% approximation of type P(i) = a*i^b


