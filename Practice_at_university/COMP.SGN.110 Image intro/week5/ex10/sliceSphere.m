function It = sliceSphere(I,dist)
 p = impixel(I);
 if size(p,1)>1
    q = p(length(p),:);
 else
     q =p;
 end
 [m,n,p]=size(I);
 I = double(I);
 r = I(:,:,1);
 g = I(:,:,2);
 b = I(:,:,3);
 P = zeros([m,n]);
 for i = 1:m
   for j = 1:n 
       
       P(i,j)=(((r(i,j)-q(1))^2+ (g(i,j)-q(2))^2+ abs(b(i,j)-q(3))^2)<=dist^2);
     
       
            
   end
    
 end
It = double(I).*P;
It = uint8(It);
end
