function [train_error, test_error, train_accuracy, test_accuracy, Total_NN_size, test_accuracy_Noisy_DCS]=...
    DCS(X_train, T_train, X_test, T_test, g, NumNodes, lam, mu, kmax, SNR_Test)
%%  Name:   PLN
%
%   Inputs:
%   X_train:        Data matrix of the training set with P rows, where each column contains
%                   one sample of dimension P (refer to "Load_dataset.m" for more details)
%   T_train:        Target matrix of the training set with Q rows, where each
%                   column containg the one target of dimension Q (refer to "Load_dataset.m" for more details)
%   X_test:         Data matrix of the testing set with P rows, where each column contains
%                   one sample of dimension P (refer to "Load_dataset.m" for more details)
%   T_test:         Target matrix of the testing set with Q rows, where each
%                   column containg the one target of dimension Q (refer to "Load_dataset.m" for more details)
%   g:              a PP-holding non-linear function such as RLU or leaky-RLU
%   NumNodes:       Matrix containing the number of nodes in each layer (each element MUST be >= 2Q)
%                   Row i contains the number nodes of layer i on which we want to sweep, and it MUST be increasing
%   eps_o:          The regularization constant of matrix O which is equal to alpha*sqrt(2*Q), alpha MUST be >= 1
%   mu:             the parameter mu of ADMM which controls the convergence speed
%   kmax:           maximum number of iteration in ADMM algorithm
%   lam:            lagrangian multiplier of the regularized least-square in the first layer
%   eta_n:          NME threshold for adding new nodes to the network
%   eta_l:          NME threshold for adding new layer to the network
%   First_Block:    Represents the choice of the algorithm in the first sublayer which in our case is 'LS'
%
%   Outputs:
%   train_error:    The training NME in db scale
%   test_error:     The testing NME in db scale
%   train_accuracy: The trainging accuracy
%   test_accuracy:  The testing accuracy
%   Total_NN_size:  The squence of total number of random nodes in the network at the time of training
%   NumNode_opt:    The optimum number of random nodes derived by PLN in each layer
%
%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Paper:          Progressive Learning for Systematic Design of Large Neural Network
%   Authors:        Saikat Chatterjee, Alireza M. Javid, Mostafa Sadeghi, Partha P. Mitra, Mikael Skoglund
%   Organiztion:    KTH Royal Institute of Technology
%   Contact:        Alireza M. Javid, almj@kth.se
%   Website:        www.ee.kth.se/reproducible/
%
%   ***September 2017***

%%
% MyCoef=Coef;

Pi=size(X_train,1);  %   Data Dimension
Q=size(T_train,1);
N=size(X_train,2);
Nt=size(X_test,2);

size_counter=0;
train_error=[];
test_error=[];
test_accuracy=[];
train_accuracy=[];
Total_NN_size=[];   %   the sequence of total number of random nodes in the network

test_error_Noisy_DCS=[];
test_accuracy_Noisy_DCS=[];

%   Initializing the algorithm for the first time
Yi=X_train;
Yi_test=X_test;
%%%%%%%%

W_cell=cell(1,length(NumNodes)+1);

layer=0;
while layer<length(NumNodes)
    layer=layer+1;
    ni=NumNodes(layer);
    
    if layer==1
        Ri=2*rand(ni-2*Q, Pi)-1;     %   Generating the random matrix R
                Zi=Ri*Yi;
                Zi_test=Ri*Yi_test;
        
%         Zi=dct(Yi,ni-2*Q);                              %   For using DCT
%         Zi_test=dct(Yi_test,ni-2*Q);
    else
        Ri=2*rand(Pi, Pi)-1;                      %   Generating the random matrix R
                Zi=Ri*Yi;
                Zi_test=Ri*Yi_test;
        
%         Zi=dct(Yi,Pi);                                      %   For using DCT
%         Zi_test=dct(Yi_test,Pi);
    end
    %         Ri=dftmtx(Pi);
    %         Ri=2*rand(Pi, Pi)-1;
    
    
    
    if layer>=1                              %%%%%%%%%%      layer >1  if ELM in the first layer, otherwise, layer>=1
        MyTemp=[eye(size(Zi,1));-eye(size(Zi,1))];
        Zi=MyTemp*Zi;
        Zi_test=MyTemp*Zi_test;
    end
    
    if layer==1
        W_cell{layer}=Ri;
    else
        W_cell{layer}=MyTemp*Ri;
    end
    
    Yi=g(Zi);
    Yi_test=g(Zi_test);
    
    % % % % % % % % % % % % % % % % % % % % % %
%     % % % % % % % % % % % % % % % % % % % % % %     For ELM in the first layer
%         if layer==1
%             [Oi,~]=LS( Yi, T_train, [], [], lam, 'No');
%         else
% %             eps_o=norm(Oi * ((Ri' * Ri) \ Ri') * [eye(size(Ri,1)),-eye(size(Ri,1))],'fro')
%             eps_o=sqrt(2)*norm(Oi,'fro')                        %%%%%%%%%%%%%%%%%%%%    if we use DCT
%             
%             Oi=LS_ADMM( T_train, Yi, eps_o, mu, kmax);    %   The ADMM solver for constrained least square
%         end
%     % % % % % % % % % % % % % % % % % % % % % %
%     % % % % % % % % % % % % % % % % % % % % % %
    
    % % % % % % % % % % % % % % % % % % % % % %
    % % % % % % % % % % % % % % % % % % % % % %     For LS in the first layer
    if layer==1
        [Oi,~]=LS( X_train, T_train, [], [], lam, 'No');
        t_hati=Oi*X_train;
        t_hati_test=Oi*X_test;
        
        train_error=[train_error,Calculate_error(T_train,t_hati)];
        train_accuracy=[train_accuracy,Calculate_accuracy(T_train,t_hati)];
        test_error=[test_error,Calculate_error(T_test,t_hati_test)];
        test_accuracy=[test_accuracy,Calculate_accuracy(T_test,t_hati_test)];
        Total_NN_size=[Total_NN_size, size_counter];
    end
    
        eps_o=norm(Oi * ((Ri' * Ri) \ Ri') * [eye(size(Ri,1)),-eye(size(Ri,1))],'fro')   %%%      if we use Random
%     eps_o=sqrt(2)*norm(Oi,'fro')                        %%%%%%%%%%%%%%%%%%%%    if we use DCT
    
    Oi=LS_ADMM( T_train, Yi, eps_o, mu, kmax);    %   The ADMM solver for constrained least square
    % % % % % % % % % % % % % % % % % % % % % %
    % % % % % % % % % % % % % % % % % % % % % %
    
    t_hati=Oi*Yi;
    t_hati_test=Oi*Yi_test;
    
    %     t_hati=softmax(t_hati);
    %     t_hati_test=softmax(t_hati_test);
    
    train_error=[train_error,Calculate_error(T_train,t_hati)];
    train_accuracy=[train_accuracy,Calculate_accuracy(T_train,t_hati)]
    test_error=[test_error,Calculate_error(T_test,t_hati_test)];
    test_accuracy=[test_accuracy,Calculate_accuracy(T_test,t_hati_test)]
    
    Pi=size(Yi,1);
    size_counter=size_counter+Pi;   % Updating the total number of random nodes at the end of each layer
    Total_NN_size=[Total_NN_size, size_counter];  %   The total number of random nodes is updating
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    %   plotting accuracy and NME in each step for train and test data (can be plotted only at the end)
    %                 figure(3)
    
    figure(1)
    subplot(2,1,1)
    plot(Total_NN_size,train_accuracy,'b','Linewidth',2);
    hold on; grid on
    plot(Total_NN_size,test_accuracy,'r:','Linewidth',2);
    ylabel('Accuracy','FontName','Times New Roman')
    xlabel('Total number of nodes','FontName','Times New Roman')
    legend('Training Accuracy','Testing Accuracy','Location','southeast')
    hold on
    subplot(2,1,2)
    plot(Total_NN_size,train_error,'b','Linewidth',2);
    hold on; grid on
    plot(Total_NN_size,test_error,'r:','Linewidth',2);
    ylabel('NME','FontName','Times New Roman')
    xlabel('Total number of nodes','FontName','Times New Roman')
    legend('Training NME','Testing NME','Location','northeast')
    hold on
    drawnow
end
W_cell{end}=Oi;


switch SNR_Test
    case {'Yes'}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Test
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        for SNR=1:50
            Yi_test=AddNoise(X_test,SNR);
            layer=0;
            while layer<length(NumNodes)
                layer=layer+1;
                Yi_test=g(W_cell{layer}*Yi_test);
            end
            t_hati_test=W_cell{end}*Yi_test;
            test_error_Noisy_DCS=[test_error_Noisy_DCS,Calculate_error(T_test,t_hati_test)];
            test_accuracy_Noisy_DCS=[test_accuracy_Noisy_DCS,Calculate_accuracy(T_test,t_hati_test)];
            
            % % % % % % % % % % % % % % % % % % % % % % % % %
            % % % % % % % % % % % % % % % % % % % % % % % % %
            %     figure(40)
            %     subplot(2,1,1)
            %     plot(test_accuracy_Noisy_DCS,'r','Linewidth',2);
            %     hold on; grid on
            %     ylabel('Accuracy','FontName','Times New Roman')
            %     xlabel('SNR','FontName','Times New Roman')
            %     hold on
            %     drawnow
            %
            %     subplot(2,1,2)
            %     plot(test_error_Noisy_DCS,'r','Linewidth',2);
            %     hold on; grid on
            %     ylabel('NME','FontName','Times New Roman')
            %     xlabel('SNR','FontName','Times New Roman')
            %     hold on
            %     drawnow
            %     % % % % % % % % % % % % % % % % % % % % % % %
            %     % % % % % % % % % % % % % % % % % % % % % % %
        end
        %save Acc_vs_SNR_DCS test_accuracy_Noisy_DCS
end

return
