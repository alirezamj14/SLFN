function [train_err, test_err, train_acc, test_acc, T_hat, Tt_hat,test_accuracy_Noisy_ELM]=ELM(X_train, T_train, X_test, T_test, lam, NumNodes,g,dim, SNR_Test)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[P,N]=size(X_train);
Nt=size(X_test,2);
test_error_Noisy_ELM=[];
test_accuracy_Noisy_ELM=[];
%% Input layer weight matrix: W

W=2*(sqrt(1))*rand(NumNodes,P+1)-1;
% W=1*randn(NumNodes,P);

Z=W*[X_train;ones(1,N)];
Zt=W*[X_test;ones(1,Nt)];

% Z=W*X;
% Zt=W*Xt;

% Nt=size(Xt,2);
% [Z,Zt]=MyDCT([X;ones(1,N)],[Xt;ones(1,Nt)],dim);

%% Hidden neorons

Y=g(Z);
Yt=g(Zt);
% Y=X;
% Yt=Xt;

% % Output layer weight matrix: O
if NumNodes < N
    O=(T_train*Y')/(Y*Y'+lam*eye(size(Y,1)));
else
    O=(T_train/(Y'*Y+lam*eye(size(Y,2))))*Y';
end

norm(O,'fro')
T_hat=O*Y;
Tt_hat=O*Yt;

% Test error and accuracy
test_acc=Calculate_accuracy(T_test,Tt_hat);
train_acc=Calculate_accuracy(T_train,T_hat);
test_err=Calculate_error(T_test,Tt_hat);
train_err=Calculate_error(T_train,T_hat);

switch SNR_Test
    case {'Yes'}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for SNR=1:50
    Input=AddNoise(X_test,SNR);
    t_hati_test=O*g(W*[Input;ones(1,Nt)]);
    
    test_error_Noisy_ELM=[test_error_Noisy_ELM,Calculate_error(T_test,t_hati_test)];
    test_accuracy_Noisy_ELM=[test_accuracy_Noisy_ELM,Calculate_accuracy(T_test,t_hati_test)];
    
    % % % % % % % % % % % % % % % % % % % % % % % % %
    % % % % % % % % % % % % % % % % % % % % % % % % %
%     figure(40)
%     subplot(2,1,1)
%     plot(test_accuracy_Noisy_ELM,'g','Linewidth',2);
%     hold on; grid on
%     ylabel('Accuracy','FontName','Times New Roman')
%     xlabel('SNR','FontName','Times New Roman')
%     hold on
%     drawnow
%     
%     subplot(2,1,2)
%     plot(test_error_Noisy_ELM,'g','Linewidth',2);
%     hold on; grid on
%     ylabel('NME','FontName','Times New Roman')
%     xlabel('SNR','FontName','Times New Roman')
%     hold on
%     drawnow
    %     % % % % % % % % % % % % % % % % % % % % % % %
    %     % % % % % % % % % % % % % % % % % % % % % % %
end

% save Acc_vs_SNR_ELM test_accuracy_Noisy_ELM
end

return
