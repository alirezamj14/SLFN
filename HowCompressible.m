function r=HowCompressible(X)
% % % %         The input is treated column-wise

% epsilon=0.01;
% K=size(X,1);
% N=size(X,2);
% Sorted_abs_x=sort(abs(X),'descend');
% C=Sorted_abs_x(1,:);
% m=(1:K)';
% M=m(:,ones(1,N));
% q=0;
% Mq=M.^(-q)*diag(C);
% while  ~isempty(find(Mq > Sorted_abs_x, 1))
%     q=q+epsilon;
%     Mq=M.^(-q)*diag(C);
% end
% r=q;


epsilon=0.1;
K=size(X,1);
N=size(X,2);
r=zeros(1,N);
for i=1:N
    Sorted_abs_x=transpose(sort(abs(X(:,i)),'descend'));
    C=max(Sorted_abs_x);
    q=0;
    while Sorted_abs_x <= C * (1:K).^(-q)
        q=q+epsilon;
    end
    r(i)=q;
end
% figure(30);plot(r)
r=sum(r)/N;
return
