function re = Decoder(ls,lsign,p)
%DECODER Summary of this function goes here
%   Detailed explanation goes here
re=zeros([1,length(lsign)]);
i=1;

j=1;


while i<length(ls)-p
    ind=find(ls(i+p+1:end)==0,1,"first");
   
    m = ls(i+1:i+p);
    
    n = (ind-1)*2^p+sum(m.*(2.^(size(m,2)-1:-1:0)),2);
    re(j)=n;
    j=j+1;
    i = i+p+ind+1;
end
lsign(lsign==1)=1;
lsign(lsign==0)=-1;
re = re.*lsign;
end

