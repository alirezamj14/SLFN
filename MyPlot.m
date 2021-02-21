function MyPlot(Z,Color,num)
Z_sorted=sort(abs(Z),'descend');
Z_mean=mean(Z,2);
figure(num)
plot(Z_mean,Color)
grid on
hold on
drawnow
return
