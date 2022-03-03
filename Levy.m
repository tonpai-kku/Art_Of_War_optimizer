% Levy flights by Mantegna's algorithm
function levyNum=Levy(m,n)
beta=3/2;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
levyNum = zeros(m,n);
for j=1:m
    u=randn(1,n)*sigma;
    v=randn(1,n);
    levyNum(j,:)=u./abs(v).^(1/beta);
end
end