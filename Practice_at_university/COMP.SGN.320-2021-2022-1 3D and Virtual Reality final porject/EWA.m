function vt = EWA(buffer)
%EWA exponentially weighted average smooth the cord
beta = 0.9;
ee = length(buffer)-1:-1:0;
temp = beta.^ee;
vt = (1-beta)*buffer*temp';
vt = vt/(1-beta^length(buffer));
end

