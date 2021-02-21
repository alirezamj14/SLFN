function [Z,Z_test]=MyDCT(Y,Y_test,dim)

Z=[];
% P=size(Y,1);
% if P <=dim
%     Z=dct(Y,dim);
%     Z_test=dct(Y_test,dim);
% else
%     Z=dct(Y);
%     Z_test=dct(Y_test);
%     Z=Z(1:dim,:);
%     Z_test=Z_test(1:dim,:);
% end

if isempty(Y)
    if dim==0
        Z_test=[];
    else
        Z_test=dct(Y_test,dim);
    end
else
    if dim==0
        Z=[];
        Z_test=[];
    else
        Z=dct(Y,dim);
        Z_test=dct(Y_test,dim);
    end
end
return
