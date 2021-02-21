close all
addpath(genpath('Results'));
FontSize=23;
% 
Fig_a=hgload('accuracy_Letter.fig');
Fig_b=hgload('NumNodes_Letter.fig');
Fig_c=hgload('accuracy_MNIST.fig');
Fig_d=hgload('NumNodes_MNIST.fig');
% Fig_e=hgload('accuracy_NORB.fig');
% Fig_f=hgload('accuracy_Shuttle.fig');
% Fig_g=hgload('accuracy_MNIST.fig');
% Fig_h=hgload('accuracy_CIFAR10.fig');
% Prepare subplots
figure(9)
h(1)=subplot(1,4,1);
grid on
box on
axis tight
ylabel('Accuracy','FontName','Times New Roman','FontWeight','normal')
xlabel('Total number of random nodes','FontName','Times New Roman','FontWeight','normal')
title('(a) Letter','FontName','Times New Roman','FontWeight','normal')
legend(h(1),'Training Accuracy','Testing Accuracy','Location','southeast')
han = gca(figure(9));
set(han,'fontsize',FontSize,'FontName','Times New Roman','LineWidth',1,'defaultLineLineWidth',1);
han.XColor = 'k'; % Red
han.YColor = 'k'; % Blue
h(2)=subplot(1,4,2);
grid on
box on
axis tight
ylabel('Number of Neurons','FontName','Times New Roman','FontWeight','normal')
xlabel('Layer Number','FontName','Times New Roman','FontWeight','normal')
title('(b) Letter','FontName','Times New Roman','FontWeight','normal')
han = gca(figure(9));
set(han,'fontsize',FontSize,'FontName','Times New Roman','LineWidth',1,'defaultLineLineWidth',1);
han.XColor = 'k'; % Red
han.YColor = 'k'; % Blue
h(3)=subplot(1,4,3);
grid on
box on
axis tight
ylabel('Accuracy','FontName','Times New Roman','FontWeight','normal')
xlabel('Total number of random nodes','FontName','Times New Roman','FontWeight','normal')
title('(c) MNIST','FontName','Times New Roman','FontWeight','normal')
legend(h(3),'Training Accuracy','Testing Accuracy','Location','southeast')
han = gca(figure(9));
set(han,'fontsize',FontSize,'FontName','Times New Roman','LineWidth',1,'defaultLineLineWidth',1);
han.XColor = 'k'; % Red
han.YColor = 'k'; % Blue
h(4)=subplot(1,4,4);
grid on
box on
axis tight
ylabel('Number of Neurons','FontName','Times New Roman','FontWeight','normal')
xlabel('Layer Number','FontName','Times New Roman','FontWeight','normal')
title('(d) MNIST','FontName','Times New Roman','FontWeight','normal')
han = gca(figure(9));
set(han,'fontsize',FontSize,'FontName','Times New Roman','LineWidth',1,'defaultLineLineWidth',1);
han.XColor = 'k'; % Red
han.YColor = 'k'; % Blue
% h(5)=subplot(2,4,5);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of random nodes','FontName','Times New Roman')
% title('(b) Satimage','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% h(6)=subplot(2,4,6);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of random nodes','FontName','Times New Roman')
% title('(d) Letter','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% h(7)=subplot(2,4,7);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of random nodes','FontName','Times New Roman')
% title('(f) Shuttle','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% h(8)=subplot(2,4,8);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of random nodes','FontName','Times New Roman')
% title('(h) CIFAR10','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% Paste figures on the subplots
copyobj(allchild(get(Fig_a,'CurrentAxes')),h(1));
copyobj(allchild(get(Fig_b,'CurrentAxes')),h(2));
copyobj(allchild(get(Fig_c,'CurrentAxes')),h(3));
copyobj(allchild(get(Fig_d,'CurrentAxes')),h(4));
% copyobj(allchild(get(Fig_e,'CurrentAxes')),h(3));
% copyobj(allchild(get(Fig_f,'CurrentAxes')),h(7));
% copyobj(allchild(get(Fig_g,'CurrentAxes')),h(4));
% copyobj(allchild(get(Fig_h,'CurrentAxes')),h(8));
% Add legends

set(gcf, 'Position',  [100, 100, 2500, 520])

% legend(h(2),'Training Accuracy','Testing Accuracy','Location','southeast')

% legend(h(4),'Training Accuracy','Testing Accuracy','Location','southeast')
% legend(h(5),'Training Accuracy','Testing Accuracy','Location','southeast')
% legend(h(6),'Training Accuracy','Testing Accuracy','Location','southeast')
% legend(h(7),'Training Accuracy','Testing Accuracy','Location','southeast')
% legend(h(8),'Training Accuracy','Testing Accuracy','Location','southeast')

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% 
% close all
% addpath(genpath('Results\DCS'));
% 
% Fig_a=hgload('Letter-InitLS.fig');
% Fig_b=hgload('Shuttle-InitLS.fig');
% Fig_c=hgload('MNIST-InitLS.fig');
% Fig_d=hgload('Letter-InitELM.fig');
% Fig_e=hgload('Shuttle-InitELM.fig');
% Fig_f=hgload('MNIST-InitELM.fig');
% 
% % Prepare subplots
% figure(9)
% h(1)=subplot(2,3,1);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of neurons','FontName','Times New Roman')
% title('(a) Letter','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% h(2)=subplot(2,3,2);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of neurons','FontName','Times New Roman')
% title('(b) Shuttle','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% h(3)=subplot(2,3,3);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of neurons','FontName','Times New Roman')
% title('(c) MNIST','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% h(4)=subplot(2,3,4);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of neurons','FontName','Times New Roman')
% title('(d) Letter','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% h(5)=subplot(2,3,5);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of neurons','FontName','Times New Roman')
% title('(e) Shuttle','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% h(6)=subplot(2,3,6);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of neurons','FontName','Times New Roman')
% title('(f) MNIST','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% % Paste figures on the subplots
% copyobj(allchild(get(Fig_a,'CurrentAxes')),h(1));
% copyobj(allchild(get(Fig_b,'CurrentAxes')),h(2));
% copyobj(allchild(get(Fig_c,'CurrentAxes')),h(3));
% copyobj(allchild(get(Fig_d,'CurrentAxes')),h(4));
% copyobj(allchild(get(Fig_e,'CurrentAxes')),h(5));
% copyobj(allchild(get(Fig_f,'CurrentAxes')),h(6));
% % Add legends

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % close all
% addpath(genpath('Results\DCS'));
% 
% % Fig_a=hgload('Letter-InitLS-DCT.fig');
% % Fig_b=hgload('Shuttle-InitLS-DCT.fig');
% % Fig_c=hgload('MNIST-InitLS-DCT.fig');
% Fig_a=hgload('Letter-InitELM-DCT.fig');
% Fig_b=hgload('Shuttle-InitELM-DCT.fig');
% Fig_c=hgload('MNIST-InitELM-DCT.fig');
% 
% % Prepare subplots
% figure(9)
% h(1)=subplot(1,3,1);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of neurons','FontName','Times New Roman')
% legend('Training Accuracy','Testing Accuracy','Location','southeast')
% title('(a) Letter','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% h(2)=subplot(1,3,2);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of neurons','FontName','Times New Roman')
% legend('Training Accuracy','Testing Accuracy','Location','southeast')
% title('(b) Shuttle','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% h(3)=subplot(1,3,3);
% grid on
% box on
% axis tight
% ylabel('Accuracy','FontName','Times New Roman')
% xlabel('Total number of neurons','FontName','Times New Roman')
% legend('Training Accuracy','Testing Accuracy','Location','southeast')
% title('(c) MNIST','FontName','Times New Roman')
% han = gca(figure(9));
% set(han,'fontsize',12,'FontName','Times New Roman');
% % Paste figures on the subplots
% copyobj(allchild(get(Fig_a,'CurrentAxes')),h(1));
% copyobj(allchild(get(Fig_b,'CurrentAxes')),h(2));
% copyobj(allchild(get(Fig_c,'CurrentAxes')),h(3));
% % Add legends