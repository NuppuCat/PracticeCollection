clear all;
close all;
clc;
%% Task (1)
img = imread('Image_3.png');
A = double(img);
h = histogram(A,NumBins=255);
s = sum(h.Values);
p = h.Values/s;
p = p(p>0);
entropy = -sum(p.*log2(p))

%% (2)
B = A(50:249,50:249);
imwrite(uint8(B),"MyImage_3.png")

%% (3)
[nr,nc] = size(B);
ymap=zeros([nr,nc]);

ymap(1,1)=B(1,1);
%preset the values of first row and col
ymap(1,2:nc)=B(1,1:nc-1);
ymap(2:nr,1) = B(1:nr-1,1);
for i = 2 : nr
    for j = 2 : nc
        yN = B(i - 1, j);
        yW = B(i, j - 1);
        yNW= B(i-1, j-1);
        ymap(i,j)=median(yN,yW,yN+yW-yNW);


    end
end

E = B-ymap;
figure();
imagesc(E)
e = E(:)';
%%
fileID = fopen('e_orin.bin','w');
fwrite(fileID,e);
fclose(fileID);
%% (4)
for p= 1:8
    s= GR_estimation(abs(e),p);
    length(s)
    p
end
% best p is 3
sign =e;
sign(sign~=0)=sign(sign~=0)./abs(sign(sign~=0));
%save sign_of_error.mat sign
%writematrix(sign,'sign_of_erro.txt','Delimiter',' ')
fileID = fopen('sign_of_erro.bin','w');
fwrite(fileID,sign,'ubit1');
fclose(fileID);
p=3;
s = GR_estimation(abs(e),p);
%writematrix(s,'GR_error.txt','Delimiter',' ')
fileID = fopen('GR_error.bin','w');
fwrite(fileID,s,'ubit1');
fclose(fileID);
%save GR_error.mat s
%% (5)
fileID = fopen('GR_error.bin','r');
ls =fread(fileID,inf,'ubit1');
fclose(fileID);

fileID = fopen('sign_of_erro.bin','r');
lsgin = fread(fileID,inf,'ubit1');
fclose(fileID);

ls=ls';
lsgin=lsgin';
re = Decoder(ls,lsgin,p);
numel(find(re~=e))
%% (6)
L= 50:50:2000;
%save the codelength of each block size
CL=[]
%for every L size
for l = L
    whole_code=[];
    optimal_p_vector=[];
    %every block need to be encode
    for i = l:l:length(e)
        svp= [];
        % calculate the block's best p value
        for p=1:8
            s = GR_estimation(e(i-l+1:i),p);
            svp = [svp length(s)];
        end
        [m,im] = min(svp);
        %save the optimal p values
        optimal_p_vector=[optimal_p_vector im];
        %save the final encode result
        whole_code=[whole_code GR_estimation(e(i-l+1:i),im)];
    end
    p_star = max(optimal_p_vector);
    % transmit p I think we can use binary code encode it directly, its
    % easily to set the first number of bits as the p*
    plength = length(de2bi(p_star))*length(optimal_p_vector);
    CL = [CL length(whole_code)+plength];
end
bpp_CL=CL/length(B(:));
%%
figure();
plot(L,bpp_CL);
clo = min(CL)
[sign,s] = image_compress(B);
length(s)
%%
dos('7z')
dos('D:\GRAM\MasterProgramme\Tampere\signal compression\project\matlab_project')
crt_dir = pwd
copyfile('D:\GRAM\MasterProgramme\Tampere\signal compression\week6 lossy and jp200\q6\7z.exe',crt_dir)
InputFileName = 'e_orin.bin'
sss=dir(InputFileName);
warning off
mkdir('./temp_dir')
ArchiveName1 = './temp_dir/Arch1.7z';
delete(ArchiveName1)
ComPPmd = ['7z a ' ArchiveName1 ' ' InputFileName ' -m0=PPmd' ];
[status, cmdout] = dos(ComPPmd);
sssa=dir(ArchiveName1);
%%
[nr nc] = size(B);
msg = ['calic8e.exe e_orin.bin ' num2str(nc) ' ' num2str(nr) ' 8 0 coded.dat'];
fid= fopen ('run.bat', 'wt');
fprintf(fid, '%s',msg);
fclose (fid);

tic;
[s1, w1] = dos(msg);
result(3) = toc;
s=dir('coded.dat');
%%

fid = fopen('img.pgm','wb');
fwrite(fid, uint8(e), 'uint8');
fclose(fid);
[~, w1] = dos('locoe.exe img.pgm');

s=dir('locoe.jls');