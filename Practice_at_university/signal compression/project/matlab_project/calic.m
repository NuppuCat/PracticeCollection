function [result]=loco(gray2)
%% gray2 is the input image with pixels in the range 0-255 
%% wite input file
fid= fopen ('MyData.raw', 'wb');
A = gray2';
A = A(:);
fwrite (fid, A, 'uchar');
% fwrite (fid, gray2, 'uchar');
fclose (fid);

%% encode
[nr nc] = size(gray2);
msg = ['calic8e.exe MyData.raw ' num2str(nc) ' ' num2str(nr) ' 8 0 coded.dat'];
fid= fopen ('run.bat', 'wt');
fprintf(fid, '%s',msg);
fclose (fid);

tic;
[s1, w1] = dos(msg);
result(3) = toc;
s=dir('coded.dat');
result(1) = s.bytes*8;
result(2) = s.bytes*8/numel(gray2);

%% decode
tic;
[s1,w1] = dos('calic8d.exe coded.dat recon.raw');
result(4) = toc;
fid= fopen ('MyData.raw', 'rb');
Himg = fread(fid, numel(gray2), 'uchar');
fclose (fid);
Himg = reshape(Himg,size(gray2'));

%% error test
er = double(Himg') - double(gray2);
result(5)=length(find(er~=0));

end
