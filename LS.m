function [W_ls,test_accuracy_Noisy_LS]=LS(X_train, T_train, X_test,T_test, lam, SNR_Test)                    
P=size(X_train,1);
N=size(X_train,2);
test_error_Noisy_LS=[];
test_accuracy_Noisy_LS=[];

if P < N
    W_ls=(T_train*X_train')/(X_train*X_train'+lam*eye(size(X_train,1)));
else
    W_ls=(T_train/(X_train'*X_train+lam*eye(size(X_train,2))))*X_train';
end

switch SNR_Test
    case {'Yes'}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for SNR=1:50
    Input=AddNoise(X_test,SNR);
    t_hati_test=W_ls*Input;
    
    test_error_Noisy_LS=[test_error_Noisy_LS,Calculate_error(T_test,t_hati_test)];
    test_accuracy_Noisy_LS=[test_accuracy_Noisy_LS,Calculate_accuracy(T_test,t_hati_test)];
    
    % % % % % % % % % % % % % % % % % % % % % % % % %
    % % % % % % % % % % % % % % % % % % % % % % % % %
%     figure(40)
%     subplot(2,1,1)
%     plot(test_accuracy_Noisy_LS,'b','Linewidth',2);
%     hold on; grid on
%     ylabel('Accuracy','FontName','Times New Roman')
%     xlabel('SNR','FontName','Times New Roman')
%     hold on
%     drawnow
%     
%     subplot(2,1,2)
%     plot(test_error_Noisy_LS,'b','Linewidth',2);
%     hold on; grid on
%     ylabel('NME','FontName','Times New Roman')
%     xlabel('SNR','FontName','Times New Roman')
%     hold on
%     drawnow
    %     % % % % % % % % % % % % % % % % % % % % % % %
    %     % % % % % % % % % % % % % % % % % % % % % % %
end

% save Acc_vs_SNR_LS test_accuracy_Noisy_LS
end

end
