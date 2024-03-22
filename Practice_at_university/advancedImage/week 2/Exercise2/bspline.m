function h = bspline(x,n)
h=x;
% B-spline function of nth order
if n==0
    h(h<-1/2)=0;
    h(h>=1/2)=0;
    h(h~=0)=1;
elseif n==1
    abh=abs(h);
    h(abh>=0 & abh<1)=1-abh(abh>=0 & abh<1);
    h(abh>1)=0;
else
    abh=abs(h);
    h(abh>=0 & abh<1)=2/3-0.5*abh(abh>=0 & abh<1).^2.*(2-abh(abh>=0 & abh<1));
    h(abh>=1 & abh<2) = 1/6*(2-abh(abh>=1 & abh<2)).^3;
    h(abh>=2)=0;
% Write your own implementation here

end