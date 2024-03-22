function [uv] = Project3DTo2D2(XYZ_data, K, R, T)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    XYZ_data = XYZ_data-T;
    fx = abs(T(3))*K(1,1)*2;
    fy = abs(T(3))*K(2,2)*2;
    
    cx = T(1)*(K(1,1)*2)+K(1,3);
    cy = T(2)*K(2,2)*2;
    
    
    U = fx*XYZ_data(1,:)./XYZ_data(3,:)+cx;
    V = fy*XYZ_data(2,:)./XYZ_data(3,:)+cy;
    uv = [U;V];
end
