%%  Name:   PLN_Performance
%
%   Generating the performance results of PLN shown in Table III, Table V
%
%   Data:   Simulated data set generated from datasets mentioned in the paper
%
%   Output: Mean and standard deviation of NME and accuracy over multiple
%           trials of PLN for classification and regression datasets, as
%           well as the running time of the PLN
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Paper:          Progressive Learning for Systematic Design of Large Neural Network
%   Authors:        Saikat Chatterjee, Alireza M. Javid, Mostafa Sadeghi, Partha P. Mitra, Mikael Skoglund
%   Organiztion:    KTH Royal Institute of Technology
%   Contact:        Alireza M. Javid, almj@kth.se
%   Website:        www.ee.kth.se/reproducible/
%
%   ***September 2017***


%% begining of the simulation

clc; clear variables; clear global;
% close all;

dataset_dir = 'C:\\Nobackup\almj\Dropbox\Database\mat files';
addpath(genpath(dataset_dir));

addpath(genpath('/home/almj/Database/mat files'))
addpath(genpath('Datasets'));
addpath(genpath('Functions'));

a_leaky_RLU=0;      %   set to a small non-zero value if you want to test leaky-RLU
g=@(x) x.*(x >= 0)+a_leaky_RLU*x.*(x < 0);
% sig=@(x) x.*(x >= 0)+a_leaky_RLU*x.*(x < 0);
sig=@(x) 1./(1+exp(-x));

%%  Choosing a dataset
% Choose one of the following datasets:

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % %       DCS       % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% Database_name='Letter';                                lam=1e-10;       mu=1e-7;           kmax=100;       nmax=1000;            lmax=3;        Delta=50;          alpha=2;       eta_n=0.005;        eta_l=0.1;
% Database_name='Shuttle';                              lam=1e-10;        mu=1e-7;         kmax=100;       nmax=1000;            lmax=3;        Delta=50;          alpha=2;       eta_n=0.005;        eta_l=0.1;
 Database_name='MNIST';                               lam=1e-10;        mu=1e-7;         kmax=100;       nmax=1000;            lmax=5;        Delta=50;          alpha=2;       eta_n=0.005;        eta_l=0.1;
% Database_name='CIFAR-10';                           lam=1e8;       mu=1e7;         kmax=100;       nmax=250;            lmax=5;        Delta=50;          alpha=2;       eta_n=0.005;        eta_l=0.1;
% Database_name='CIFAR-100';                         lam=1e10;        mu=1e0;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=10;        Delta=50;
% Database_name='SVHN';                                lam=1e10;       mu=1e4;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % %       PLN      % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%
% Database_name='Vowel';                                 lam=1e2;        mu=1e3;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;
% Database_name='Satimage';                           lam=1e6;        mu=1e5;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;        lmax=20;        Delta=50;
% Database_name='Caltech101';                        lam=5e0;        mu=1e-2;        kmax=100;       alpha=3;        nmax=1000;          eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;
% Database_name='Letter';                                  lam=1e-5;       mu=1e4;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;      lmax=20;        Delta=50;
% Database_name='NORB';                                 lam=1e2;        mu=1e2;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;        lmax=20;        Delta=50;
% Database_name='Shuttle';                               lam=1e5;        mu=1e4;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;         eta_l=0.1;        lmax=20;        Delta=50;
% Database_name='MNIST';                               lam=1e0;        mu=1e5;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;       lmax=20;        Delta=50;
% Database_name='CIFAR-10';                           lam=1e8;       mu=1e3;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;       lmax=20;      Delta=50;
% Database_name='CIFAR-100';                         lam=1e8;        mu=1e0;         kmax=100;       alpha=2;        nmax=1000;      eta_n=0.005;        eta_l=0.1;       lmax=20;      Delta=50;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % %       ELM      % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% Database_name='Vowel';                                lam=1e2;        nmax=2000;            %   for DCT:   lam=1e-1
% Database_name='Satimage';                           lam=1e6;        nmax=4000;      %   for DCT:   lam=1e2
% Database_name='Caltech101';                        lam=5e0;        mu=1e-2;        kmax=100;       alpha=3;        nmax=20;          eta_n=1e-4;        eta_l=1e-4;        lmax=20;        Delta=5;
% Database_name='Letter';                                lam=1e-5;       mu=1e4;         kmax=100;       alpha=2;        nmax=1000;      eta_n=1e-4;        eta_l=1e-4;         lmax=20;        Delta=500;
% Database_name='NORB';                                 lam=1e2;        mu=1e2;         kmax=100;       alpha=2;        nmax=4000;      eta_n=1e-4;        eta_l=1e-4;        lmax=20;        Delta=500;
% Database_name='Shuttle';                              lam=1e5;        mu=1e4;         kmax=100;       alpha=2;        nmax=1000;       eta_n=1e-4;        eta_l=1e-4;        lmax=20;        Delta=500;
% Database_name='MNIST';                               lam=1e3;        nmax=1000;      %   for DCT:   lam=1e1
% Database_name='CIFAR-10';                           lam=1e-2;       mu=1e7;         kmax=100;       alpha=2;        nmax=1000;       eta_n=1e-4;        eta_l=1e-4;       lmax=200;      Delta=100;
% Database_name='CIFAR-100';                         lam=1e8;        mu=1e0;         kmax=100;       alpha=2;        nmax=1000;       eta_n=1e-4;        eta_l=1e-4;;       lmax=100;      Delta=100;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

MyFunc=cell(1,1);
MyFunc{1}=str2func('RandProj');
% MyFunc{1}=str2func('MyDCT');
% MyFunc{3}=str2func('MyHadamard');

%   Loading the dataset
[X_train,T_train,X_test,T_test]=Load_dataset(Database_name);

% X_train=zscore(X_train);
% X_test=zscore(X_test);

P=size(X_train,1);
Q=size(T_train,1);  %   Target Dimension
% nmax=8*Q;
trialNum=30;

% The perfomance measures we are interested in
NumNodes_opt_PLN=zeros(100,trialNum);
train_error_PLN=[];
test_error_PLN=[];
accuracy_PLN=[];
train_accuracy_PLN=[];
time_PLN=[];

% %   Generating the set of nodes in each layer
lmax=20;
% nmax=1000;
NumNodes_min=2*Q+nmax;
NumNodes_max=2*Q+nmax;
temp=NumNodes_min:Delta:NumNodes_max;
ind=ones(lmax,1);
NumNodes=temp(ind,:);

% NumNodes=[240:100:1040,2040:1000:12040];

eps_o=alpha*sqrt(2*Q)  %   the regularization constant
First_Block='LS';

learning_rate=10^(-4);
lr_decrease_enabled=0;
num_of_epoch_max=10000;
error_point_step=1;

Coef=4;

% % Finding the optimum number of random nodes in each layer
% [train_error, test_error, train_accuracy, test_accuracy, Total_NN_size, NumNodes_opt]=PLN(X_train, T_train,...
%     X_test, T_test, g, NumNodes, eps_o, mu, kmax, lam, eta_n, eta_l, First_Block,Coef);

sweep=10.^(-10:10);
% title(Database_name)

My_Bag=[];
% Running the network with the optimum number nodes in each layer derived above

Acc_SNR_DCS_i=[];
Acc_SNR_ELM_i=[];
Acc_SNR_LS_i=[];

j=1;
for i=1:1
    i
    for nmax=nmax
        %         accuracy_PLN=[];
        %         train_accuracy_PLN=[];
        for lam=lam
            %             tic
            %   Loading the dataset each time to reduce the effect of random partitioning in some of the datasets
            [X_train,T_train,X_test,T_test]=Load_dataset(Database_name);
            %             X_train=zscore(X_train);
            %             X_test=zscore(X_test);
            %                         eps_o=alpha*sqrt(2*Q);  %   the regularization constant
            
            [train_error, test_error, train_accuracy, test_accuracy, Total_NN_size,test_accuracy_Noisy_DCS]=DCS...
                (X_train, T_train, X_test, T_test, g, NumNodes, lam, mu, kmax, 'No');
%             Acc_SNR_DCS_i=[Acc_SNR_DCS_i;test_accuracy_Noisy_DCS];
%             
%             [train_error, test_error, train_accuracy, test_accuracy, T_hat, Tt_hat,test_accuracy_Noisy_ELM]=ELM(X_train, T_train, X_test, T_test, lam, nmax,g,[],'Yes');
%             Acc_SNR_ELM_i=[Acc_SNR_ELM_i;test_accuracy_Noisy_ELM];
%             
%             [~,test_accuracy_Noisy_LS]=LS(X_train, T_train, X_test, T_test, lam,'Yes');
%             Acc_SNR_LS_i=[Acc_SNR_LS_i;test_accuracy_Noisy_LS];
            
            %                         [X_train2,X_test2]=MyEdge(X_train,X_test);
            
%                         [train_error, test_error, train_accuracy, test_accuracy, Total_NN_size, MyTransforms,MyRatio]=PLN(X_train, T_train,...
%                             X_test, T_test, g, g, NumNodes, eps_o, mu, 100, lam, 0, 0, First_Block,MyFunc);
            %             My_Bag=[My_Bag,MyTransforms];
            %             toc
            
            accuracy_PLN=[accuracy_PLN,test_accuracy(end)];
            train_accuracy_PLN=[train_accuracy_PLN,train_accuracy(end)];
            
            %             time_PLN=[time_PLN,PLN_time];
            %
            figure(2)
            %             subplot(2,1,1)
            plot(accuracy_PLN,'r','Linewidth',2)
            hold on
            grid on
            plot(train_accuracy_PLN,'g','Linewidth',2)
            ylabel('Classification Accuracy')
            %             subplot(2,1,2)
            %             plot(train_error_PLN,'g','Linewidth',2)
            %             hold on
            %             grid on
            %             plot(test_error_PLN,'r','Linewidth',2)
            %             hold off
            %             xlabel('Number of hidden neurons (N)')
            ylabel('Accuracy')
            legend('Test Accuracy','Train Accuracy','location','southeast')
            title(['Regularized ELM, ',Database_name])
            drawnow
            
            j=j+1;
        end
    end
end

% Acc_SNR_DCS=mean(Acc_SNR_DCS_i,1);
% Acc_SNR_ELM=mean(Acc_SNR_ELM_i,1);
% Acc_SNR_LS=mean(Acc_SNR_LS_i,1);
% 
% save Acc_vs_SNR Acc_SNR_DCS Acc_SNR_ELM Acc_SNR_LS

% han = gca(figure(1));
% set(han,'fontsize',12,'FontName','Times New Roman');
% axis tight
% box on
% Calculating the average and standard deviation over multiple trials
mean_train_error=mean(train_error_PLN);
mean_test_error=mean(test_error_PLN);
mean_accuracy=mean(accuracy_PLN);
mean_time=mean(time_PLN);

std_train_e=std(train_error_PLN);
std_test_e=std(test_error_PLN);
std_accuracy=std(accuracy_PLN);

% Displaying the results of PLN
disp(['Performance results of "',Database_name,'" dataset:'])

disp(['Train NME = ',num2str(mean_train_error),'+',num2str(std_train_e),...
    ', Test NME = ',num2str(mean_test_error),'+',num2str(std_test_e),...
    ', Test accuracy = ',num2str(100*mean_accuracy),'+',num2str(100*std_accuracy),...
    ', Running Time = ',num2str(mean_time)])

% save Letter_performance train_error test_error train_accuracy test_accuracy Total_NN_size
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% nn=nnz(sum(NumNodes_opt_PLN,2));
% figure(2);
% [ii,~,v] = find(NumNodes_opt_PLN);
% MyMean = accumarray(ii,v,[],@mean)';
% [ii,~,v] = find(NumNodes_opt_PLN);
% Mystd = accumarray(ii,v,[],@std)';
% shadedErrorBar(1:nn,NumNodes_opt_PLN(1:nn,:)',MyMean,Mystd,{@mean,@std});
% hold on
% grid on
% plot(1:nn,NumNodes_opt_PLN(1:nn,:)','*','color',[0.5,0.5,0.95])
% box on
% axis tight
% xlabel('Layer Number')
% ylabel('Number of neurons')
% han = gca(figure(2));
% set(han,'fontsize',12,'FontName','Times New Roman');

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

axis tight
han = gca(figure(9));
FontSize=12;
% set(han,'fontsize',12,'FontName','Times New Roman');
set(gcf, 'Position',  [100, 100, 3000, 580])
set(han,'fontsize',FontSize,'FontName','Times New Roman','LineWidth',1,'defaultLineLineWidth',1);


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% plot(Acc_SNR_DCS,'r','Linewidth',2);
% hold on; grid on
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('SNR','FontName','Times New Roman')
% plot(Acc_SNR_ELM,'g','Linewidth',2);
% plot(Acc_SNR_LS,'b','Linewidth',2);
% legend('DCS','ELM','LS','Location','southeast')
