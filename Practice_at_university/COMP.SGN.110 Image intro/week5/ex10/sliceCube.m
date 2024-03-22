function It = sliceCube(I,dist)
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
 P = zeros(m,n);
 It = I;
 for i = 1:m
   for j = 1:n 
       d = abs(r(i,j)-q(1))+ abs(g(i,j)-q(2))+ abs(b(i,j)-q(3));
       if d<=dist
 %       It(i,j,1)=0;
 %       It(i,j,2)=0;
 %       It(i,j,3)=0;
 %      else
%        It(i,j,1)=r(i,j);  
 %       It(i,j,2)=g(i,j); 
%        It(i,j,3)=b(i,j); 
       P(i,j)=1; 
       end
              
            
   end
    
 end
 It = It.*P;
It = uint8(It);
end
