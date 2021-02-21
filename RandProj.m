function [Z,Z_test]=RandProj(Y,Y_test,dim)

Z=[];

p=size(Y_test,1);
rng(1000)
R=2*rand(dim, p)-1;     %   Generating the random matrix R

if isempty(Y)
    Z_test=normc(R*Y_test);
else
    Z=normc(R*Y);
    Z_test=normc(R*Y_test);
end
return
