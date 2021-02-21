function [train_error, test_error, train_accuracy, test_accuracy, Total_NN_size, MyTransforms,MyRatio]=...
    PLN(X_train, T_train, X_test, T_test, g, sig, NumNodes, eps_o, mu, kmax, lam, eta_n, eta_l, First_Block,MyFunc)
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
W_cell=cell(1,size(NumNodes,1)+1);

%% First layer Block
switch First_Block
    case {'LS'}
%         Oi=LS( T_train, X_train, lam);
        [Oi,~]=LS( X_train, T_train, [], [], lam, 'No');
        train_label_firstBlock=Oi*X_train;
        test_label_firstBlock=Oi*X_test;
        train_accuracy_firstBlock=Calculate_accuracy(T_train,train_label_firstBlock);
        test_accuracy_firstBlock=Calculate_accuracy(T_test,test_label_firstBlock);
end
W_cell{1}=Oi;

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

MyTransforms=cell(size(NumNodes,1),1);
MyRatio=[];
layer=0;
while layer<size(NumNodes,1)
    layer=layer+1;
    
    Train_Error_Li=zeros(1,size(NumNodes,2));
    Test_Error_Li=zeros(1,size(NumNodes,2));
    
    for nn=1:size(NumNodes,2)
        nl=NumNodes(layer,nn);
        dim=nl-2*Q;
        
        Zi_part1=VQ*t_hati;
        Zi_part1_test=VQ*t_hati_test;
        
        if length(MyFunc)==1
            IndMax=1;
        else
            MyTemp=zeros(1,length(MyFunc));
            for i=1:length(MyFunc)
                
                [Zi_part2,Zi_part2_test]=MyFunc{i}(Yi,Yi_test,dim);
%                 Zi_part2_test=MyFunc{i}(Yi_test,dim);
%                 MyPlot(sig(Zi_part2),'b',10)
                
                Yi_temp=[g(Zi_part1);sig(Zi_part2)];
                
                Oi=LS_ADMM(T_train,Yi_temp,eps_o, mu, kmax);    %   The ADMM solver for constrained least square
%                 figure(15);imshow(Oi(:,1:100),'InitialMagnification', 1000)
                t_hati=Oi*Yi_temp;
                
                %%%%%%%%%%  Test
                %   Following the same procedure for test data
                
                Yi_test_temp=[g(Zi_part1_test);sig(Zi_part2_test)];
                t_hati_test=Oi*Yi_test_temp;
                
                MyTemp(i)=Calculate_accuracy(T_test,t_hati_test);
            end
            [~,IndMax]=max(MyTemp);         %%%%%   Finding maximum test accuracy among the bag of transform
        end
        
        MyTransforms{layer}=func2str(MyFunc{IndMax});
        
        [Zi_part2,Zi_part2_test]=MyFunc{IndMax}(Yi,Yi_test,dim);
%         Zi_part2_test=MyFunc{IndMax}(Yi_test,dim);

%         if layer==1
%             dim=dim-400;
%             Zi_part2=Zi_part2(1:dim,:);
%             Zi_part2_test=Zi_part2_test(1:dim,:);        
%         end
        
        Zi=[Zi_part1;Zi_part2];
        %     Yi_temp=g(Zi);
        Yi_temp=[g(Zi_part1);sig(Zi_part2)];
        
%         r=HowCompressible(sig(Zi_part2));
%         MyRatio=[MyRatio,r];
%         figure(20);plot(MyRatio)
%         drawnow
        
        Oi=LS_ADMM(T_train,Yi_temp,eps_o, mu, kmax);    %   The ADMM solver for constrained least square
%         figure(15);imshow(Oi(:,1:100),'InitialMagnification', 1000);drawnow
        W_cell{layer+1}=Oi;
        t_hati=Oi*Yi_temp;
        
        train_error=[train_error,Calculate_error(T_train,t_hati)];
        train_accuracy=[train_accuracy,Calculate_accuracy(T_train,t_hati)];
        Train_Error_Li(nn)=train_error(end);
        
        %%%%%%%%%%  Test
        %   Following the same procedure for test data
        
        Zi_test=[Zi_part1_test;Zi_part2_test];
        %     Yi_test_temp=g(Zi_test);
        Yi_test_temp=[g(Zi_part1_test);sig(Zi_part2_test)];
        t_hati_test=Oi*Yi_test_temp;
        
        test_error=[test_error,Calculate_error(T_test,t_hati_test)];
        test_accuracy=[test_accuracy,Calculate_accuracy(T_test,t_hati_test)];
        Test_Error_Li(nn)=test_error(end);
        
        %         size_counter=size_counter+dim;   % Updating the total number of random nodes at the end of each layer
        Total_NN_size=[Total_NN_size,size_counter+dim];  %   The total number of random nodes is updating
        
        % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
        % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
        %   plotting accuracy and NME in each step for train and test data (can be plotted only at the end)
        
        Color1='b';
        Color2='r:';
        
        figure(1)
        subplot(2,1,1)
        plot(Total_NN_size,train_accuracy,Color1,'Linewidth',2);
        hold on; grid on
        plot(Total_NN_size,test_accuracy,Color2,'Linewidth',2);
        ylabel('Accuracy','FontName','Times New Roman')
        xlabel('Total number of random nodes','FontName','Times New Roman')
        legend('Training Accuracy','Testing Accuracy','Location','southeast')
        hold on
        drawnow
        
        subplot(2,1,2)
        plot(Total_NN_size,train_error,Color1,'Linewidth',2);
        hold on; grid on
        plot(Total_NN_size,test_error,Color2,'Linewidth',2);
        ylabel('NME','FontName','Times New Roman')
        xlabel('Total number of random nodes','FontName','Times New Roman')
        legend('Training NME','Testing NME','Location','northeast')
        hold on
        drawnow
        
        %         subplot(2,1,2)
        %         plot(NumNodes(layer,1:nn)-2*Q,Train_Error_Li(1:nn),Color1,'Linewidth',2);
        %         hold on; grid on
        %         plot(NumNodes(layer,1:nn)-2*Q,Test_Error_Li(1:nn),Color2,'Linewidth',2);
        %         ylabel('NME','FontName','Times New Roman')
        %         xlabel('Total number of random nodes','FontName','Times New Roman')
        %         legend('Training NME','Testing NME','Location','northeast')
        %         hold on
        %         drawnow
        
    end
    %    updating the variables for the next layer
    Yi=Yi_temp(1:end,:);
    Yi_test=Yi_test_temp(1:end,:);
    Pi=size(Yi,1);
    size_counter=Total_NN_size(end);   % Updating the total number of random nodes at the end of each layer
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Test
% Yi_test=X_test;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [~,ZZ]=MyFunc{1}([],Yi_test,1000);
% % MyPlot(ZZ,'b',10)
% 
% test_error_Noisy=[];
% test_accuracy_Noisy=[];
% 
% for SNR=100
%     Yi_test=AddNoise(X_test,SNR);
%     
%     layer=0;
%     while layer<size(NumNodes,1)
%         layer=layer+1;
%         nl=NumNodes(layer,end);
%         dim=nl-2*Q;
%         [~,ZZ]=MyFunc{1}([],Yi_test,dim);
% %         MyPlot(ZZ,'r',10)
%         
% %         if layer==1
% %             dim=dim-400;
% %             ZZ=ZZ(1:dim,:);
% %         end
%         
%         Yi_test=[g(VQ*W_cell{layer}*Yi_test);sig(ZZ)];
%     end
%     Yi_test=W_cell{layer+1}*Yi_test;
%     
%     test_error_Noisy=[test_error_Noisy,Calculate_error(T_test,Yi_test)];
%     test_accuracy_Noisy=[test_accuracy_Noisy,Calculate_accuracy(T_test,Yi_test)];
%     
%     % % % % % % % % % % % % % % % % % % % % % % %
%     % % % % % % % % % % % % % % % % % % % % % % %
% %     figure(40)
% % %     subplot(2,1,1)
% %     plot(test_accuracy_Noisy,Color2,'Linewidth',2);
% %     hold on; grid on
% %     %                 plot(test_accuracy,'r:','Linewidth',2);
% %     ylabel('Accuracy','FontName','Times New Roman')
% %     xlabel('SNR','FontName','Times New Roman')
% %     %                 legend('Training Accuracy','Testing Accuracy','Location','southeast')
% %     hold on
% %     drawnow
%     
% %     subplot(2,1,2)
% %     plot(test_error_Noisy,'r','Linewidth',2);
% %     hold on; grid on
% %     %                 plot(test_error,'r:','Linewidth',2);
% %     ylabel('NME','FontName','Times New Roman')
% %     xlabel('SNR','FontName','Times New Roman')
% %     %                 legend('Training NME','Testing NME','Location','northeast')
% %     hold off
% %     drawnow
%     % % % % % % % % % % % % % % % % % % % % % % %
%     % % % % % % % % % % % % % % % % % % % % % % %
% end

return
