function [train_error, test_error, train_accuracy, test_accuracy, Total_NN_size, NumNode_opt]=...
    PLN_backProp(X_train, T_train, X_test, T_test, g, NumNodes, eps_o, mu, kmax, lam, eta_n, eta_l, First_Block,lr_decrease_enabled,learning_rate,num_of_epoch_max,error_point_step)
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

P=size(X_train,1);  %   Data Dimension
Q=size(T_train,1);
VQ=[eye(Q);-eye(Q)];

size_counter=0;
train_error=[];
test_error=[];
test_accuracy=[];
train_accuracy=[];
Total_NN_size=[];   %   the sequence of total number of random nodes in the network
NumNode_opt=[];     %   The set of optimum number of nodes in each layer

%% First layer Block
switch First_Block
    case {'LS'}
        [train_label_firstBlock,test_label_firstBlock,train_accuracy_firstBlock,test_accuracy_firstBlock]=LS(...
            X_train,T_train,X_test,T_test, lam);
end

train_error(1)=Calculate_error(T_train,train_label_firstBlock);
test_error(1)=Calculate_error(T_test,test_label_firstBlock);
test_accuracy(1)=test_accuracy_firstBlock;
train_accuracy(1)=train_accuracy_firstBlock;
Total_NN_size=[Total_NN_size,size_counter];     %   At this point, the total number of random nodes is zero in the network

%   Initializing the algorithm for the first time
Yi=X_train;
t_hati=train_label_firstBlock;
Pi=P;

%%%%%%%%    Test
Yi_test=X_test;
t_hati_test=test_label_firstBlock;
%%%%%%%%

Thr_l=1;    %   The flag correspoding to eta_l
layer=0;
while layer<size(NumNodes,1)
    layer=layer+1;
    
    if Thr_l==1
        Ri=2*rand(NumNodes(layer,1)-2*Q, Pi)-1;     %   Generating the random matrix R
        Zi_part1=VQ*t_hati;
        Zi_part1_test=VQ*t_hati_test;
        
        Thr_n=1;    %   The flag correspoding to eta_n
        i=0;
        while i<size(NumNodes,2)
            i=i+1;
            if i==2
                Thr_n=1;
            end
            
            if Thr_n==1
                ni=NumNodes(layer,i);
                
                Total_NN_size=[Total_NN_size,size_counter+ni-2*Q];  %   The total number of random nodes is updating
                
                if i>1
                    Ri=[Ri;2*rand(ni-NumNodes(layer,i-1), Pi)-1];   %   adding new random nodes to the network
                end
                
                Zi_part2=Ri*Yi;
                Zi_part2=normc(Zi_part2);   %   The regularization action to be done at each layer
                Zi=[Zi_part1;Zi_part2];
                Yi_temp=g(Zi);
                
                Oi=LS_ADMM(T_train,Yi_temp,eps_o, mu, kmax);    %   The ADMM solver for constrained least square
                t_hati_prev=t_hati;
                t_hati=Oi*Yi_temp;
                
                train_error=[train_error,Calculate_error(T_train,t_hati)];
                train_accuracy=[train_accuracy,Calculate_accuracy(T_train,t_hati)];
                
                %%%%%%%%%%  Test
                %   Following the same procedure for test data
                Zi_part2_test=Ri*Yi_test;
                Zi_part2_test=normc(Zi_part2_test);
                Zi_test=[Zi_part1_test;Zi_part2_test];
                Yi_test_temp=g(Zi_test);
                t_hati_test_prev=t_hati_test;
                t_hati_test=Oi*Yi_test_temp;
                
                test_error=[test_error,Calculate_error(T_test,t_hati_test)];
                test_accuracy=[test_accuracy,Calculate_accuracy(T_test,t_hati_test)];
                
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
                %   plotting accuracy and NME in each step for train and test data (can be plotted only at the end)
                subplot(2,1,1)
                plot(Total_NN_size,train_accuracy,'b','Linewidth',2);
                hold on; grid on
                plot(Total_NN_size,test_accuracy,'r:','Linewidth',2);
                ylabel('Accuracy','FontName','Times New Roman')
                xlabel('Total number of random nodes','FontName','Times New Roman')
                legend('Training Accuracy','Testing Accuracy','Location','southeast')
                hold off
                
                subplot(2,1,2)
                plot(Total_NN_size,train_error,'b','Linewidth',2);
                hold on; grid on
                plot(Total_NN_size,test_error,'r:','Linewidth',2);
                ylabel('NME','FontName','Times New Roman')
                xlabel('Total number of random nodes','FontName','Times New Roman')
                legend('Training NME','Testing NME','Location','northeast')
                hold off
                drawnow
                
                %    checking to see if any of the thresholds has been reached or not
                Thr_n=((train_error(end-1)-train_error(end))/abs(train_error(end-1)))>=eta_n;
                
                if size(NumNodes,2)==1
                    if i==1 && layer>1
                        Thr_l=((train_error(end-1)-train_error(end))/abs(train_error(end-1)))>=eta_l;
                    end
                else
                    if i==1
                        error_temp=train_error(end-1);
                    end
                end
                
            end
            
        end
        
        if size(NumNodes,2)>1
            Thr_l=((error_temp-train_error(end))/abs(error_temp))>=eta_l;
        end
        
        %    updating the variables for the next layer
        Yi_prev=Yi;
        Yi=Yi_temp;
        Yi_test_prev=Yi_test;
        Yi_test=Yi_test_temp;
        Pi=ni;
        NumNode_opt=[NumNode_opt,ni];   %   Optimum number of nodes at this layer
        size_counter=size_counter+ni-2*Q;   % Updating the total number of random nodes at the end of each layer
    end
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % BackProp % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

error_point_id = 1;
for i = 1:num_of_epoch_max
    % forward step
    Zi_part2 = Ri*Yi_prev;
    Zi_part2=normc(Zi_part2);
    Zi=[VQ*t_hati_prev;Zi_part2];
    Yi = g(Zi);
    T_cup=Oi*Yi;
    
    % error backpropagation step
    DELTA =  T_train - T_cup; % delta_k = t_k - O_k
    %     delta_change = sum(sum(abs(DELTA)));
    %     if delta_change < delta_change_threshold
    %         break;
    %     end
    Oi = Oi + learning_rate * DELTA * Yi'; % update the output weights
    OMEGA = true(size(Yi(2*Q+1:end,:)));
    OMEGA(Yi(2*Q+1:end,:) <= 0) = false; % caculate the derivative
    SIGMA = Oi(:,2*Q+1:end)' * DELTA;
    Ri = Ri + learning_rate * SIGMA .* OMEGA * Yi_prev'; % update the input weights
    
    if lr_decrease_enabled
        learning_rate = learning_rate - learning_rate_decrement;
        learning_rate = max(learning_rate,learning_rate_min);
    end
    
    % print out the progress and caculate the error and the accuracy
    if mod(i,error_point_step) == 0
        train_err_bp(error_point_id) = Calculate_error(T_train,T_cup);
        
        Zi_part2 = Ri*Yi_test_prev;
        Zi_part2=normc(Zi_part2);
        Zi=[VQ*t_hati_test_prev;Zi_part2];
        Yi = g(Zi);
        T_cup_test=Oi*Yi;
        
        test_err_bp(error_point_id) = Calculate_error(T_test,T_cup_test);
        test_acc_bp(error_point_id) = Calculate_accuracy(T_test,T_cup_test);
        
        error_point_id = error_point_id + 1;
    end
    
    subplot(2,2,1)
    plot(train_err_bp(1:error_point_id-1),'Linewidth',2)
    ylabel('Train NME')
    subplot(2,2,2)
    plot(test_err_bp(1:error_point_id-1),'Linewidth',2)
    ylabel('Test NME')
    subplot(2,2,4)
    plot(test_acc_bp(1:error_point_id-1),'Linewidth',2)
    ylabel('Test Accuracy')
    drawnow
end

% error_point_id = 1;
% for i = 1:num_of_epoch_max
%     % forward step
%     Zi_part2 = Ri*X_train;
%     Zi_part2=normc(Zi_part2);
%     Zi=[VQ*train_label_firstBlock;Zi_part2];
%     Yi = g(Zi);
%     T_cup=Oi*Yi;
%     
%     % error backpropagation step
%     DELTA =  T_train - T_cup; % delta_k = t_k - O_k
%     %     delta_change = sum(sum(abs(DELTA)));
%     %     if delta_change < delta_change_threshold
%     %         break;
%     %     end
%     Oi = Oi + learning_rate * DELTA * Yi'; % update the output weights
%     OMEGA = true(size(Yi(2*Q+1:end,:)));
%     OMEGA(Yi(2*Q+1:end,:) <= 0) = false; % caculate the derivative
%     SIGMA = Oi(:,2*Q+1:end)' * DELTA;
%     Ri = Ri + learning_rate * SIGMA .* OMEGA * X_train'; % update the input weights
%     
%     if lr_decrease_enabled
%         learning_rate = learning_rate - learning_rate_decrement;
%         learning_rate = max(learning_rate,learning_rate_min);
%     end
%     
%     % print out the progress and caculate the error and the accuracy
%     if mod(i,error_point_step) == 0
%         train_err_bp(error_point_id) = Calculate_error(T_train,T_cup);
%         
%         Zi_part2 = Ri*X_test;
%         Zi_part2=normc(Zi_part2);
%         Zi=[VQ*test_label_firstBlock;Zi_part2];
%         Yi = g(Zi);
%         T_cup_test=Oi*Yi;
%         
%         test_err_bp(error_point_id) = Calculate_error(T_test,T_cup_test);
%         test_acc_bp(error_point_id) = Calculate_accuracy(T_test,T_cup_test);
%         
%         error_point_id = error_point_id + 1;
%     end
%     
%     subplot(2,2,1)
%     plot(train_err_bp(1:error_point_id-1),'Linewidth',2)
%     ylabel('Train NME')
%     subplot(2,2,2)
%     plot(test_err_bp(1:error_point_id-1),'Linewidth',2)
%     ylabel('Test NME')
%     subplot(2,2,4)
%     plot(test_acc_bp(1:error_point_id-1),'Linewidth',2)
%     ylabel('Test Accuracy')
%     drawnow
% end

return
