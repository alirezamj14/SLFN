function error=Calculate_error(T,T_hat)
%   Calculate error
error=20*log10(norm(T-T_hat,'fro')/norm(T,'fro'));
% error=norm(T-T_hat,'fro')/size(T,2);
end