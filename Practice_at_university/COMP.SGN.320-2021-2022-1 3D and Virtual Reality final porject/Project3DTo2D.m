function [uv] = Project3DTo2D(XYZ_data, K, R, T)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fx = K(1,1);
fy = K(2,2);
cx = K(1,3);
cy = K(2,3);
uv = zeros(2, length(XYZ_data));
for n = 1:length(XYZ_data)
    translated = XYZ_data(:,n)+T;
    uv(1,n) = fx*(translated(1)/translated(3))+cx;
    uv(2,n) = fy*(translated(2)/translated(3))+cy;
end

