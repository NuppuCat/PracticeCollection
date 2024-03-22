function [sign,s] = image_compress(image)
B=image;
[nr,nc] = size(B);
ymap=zeros([nr,nc]);

ymap(1,1)=B(1,1);
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
%imagesc(E)
e = E(:)';
L=[]
for p= 1:8
    s= GR_estimation(abs(e),p);
    L=[L length(s)];
    
end
[m,p]=min(L);
% best p is 3
sign =e;
sign(sign~=0)=sign(sign~=0)./abs(sign(sign~=0));
%save sign_of_error.mat sign
%writematrix(sign,'sign_of_erro.txt','Delimiter',' ')
fileID = fopen('sign_of_erro_f.bin','w');
fwrite(fileID,sign,'ubit1');
fclose(fileID);

s = GR_estimation(abs(e),p);
%writematrix(s,'GR_error.txt','Delimiter',' ')
fileID = fopen('GR_error_f.bin','w');
fwrite(fileID,s,'ubit1');
fclose(fileID);

end

