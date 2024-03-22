function s = GR_estimation(b,p)
s=[];
for i = b
    
    m= mod(i,2^p);
    code =  int2bit(m,p);
    d= floor(i/2^p);
    
    s =[s 0 code' ones([1,d]) 0];
end


end

