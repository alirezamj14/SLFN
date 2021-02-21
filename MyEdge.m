function [Z,Z_test]=MyEdge(Y,Y_test)

Z=zeros(size(Y,1)/3,size(Y,2));
for i=1:size(Y,2)
    [temp,~]=edgecolor2(reshape(Y(:,i),[32,32,3]));
    Z(:,i)=temp(:);
end

Z_test=zeros(size(Y_test,1)/3,size(Y_test,2));
for i=1:size(Y_test,2)
    [temp,~]=edgecolor2(reshape(Y_test(:,i),[32,32,3]));
    Z_test(:,i)=temp(:);
end

return
