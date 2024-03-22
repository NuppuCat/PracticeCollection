function drawSpiral(a,q)
    theta = 0:0.01:20;
    r = a*q.^theta;
    polarplot(theta,r);

end