function [Z,Z_test]=MyHadamard(Y,Y_test)

if size(Y,1)<=1024
    Y=[Y;zeros(1024-size(Y,1),size(Y,2))];
    Y_test=[Y_test;zeros(1024-size(Y_test,1),size(Y_test,2))];
else
    Y=Y(1:1024,:);
    Y_test=Y_test(1:1024,:);
end
H=(1/sqrt(2))*hadamard(size(Y,1));
Z=H*Y;
Z_test=H*Y_test;
return
