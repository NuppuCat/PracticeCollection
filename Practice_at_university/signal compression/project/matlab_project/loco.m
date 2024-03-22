function [result]=loco(gray2)
%% gray2 is the input image with pixels in the range 0-255 
%% write input image
imwrite(uint8(gray2),'img.pgm');
fid = fopen('img.pgm','rb');
a = fread(fid);
fclose(fid);

pos = find(a==32);
a=[a(1:pos(1)-1); 13; 10; a(pos(1)+1:pos(3)-1); 13; 10; a(pos(3)+1:end)];
fid = fopen('img.pgm','wb');
fwrite(fid, uint8(a), 'uint8');
fclose(fid);

%% encode
tic;
[~, w1] = dos('locoe.exe img.pgm');
result(3) = toc;
% pos = findstr(w1, 'Time =');
% w1 = w1(pos+7:end);
% pos2 = findstr(w1, ' ');
% w1 = w1(1:pos2(1)-1);
% result(3) = str2num(w1);

s=dir('locoe.jls');
result(1) = s.bytes*8;
result(2) = s.bytes*8/numel(gray2);

%% decode
tic;
[~, w1]=dos('locod.exe locoe.jls');
result(4) = toc;
% pos = findstr(w1, 'Time =');
% w1 = w1(pos+7:end);
% pos2 = findstr(w1, ' ');
% w1 = w1(1:pos2(1)-1);
% result(4) = str2num(w1);

%% error test
A=imread('locod1.out');
er = double(A) - double(gray2);
result(5) = length(find(er~=0));

end
