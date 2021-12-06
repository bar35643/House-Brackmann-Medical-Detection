function PlotGraphs(path, fig)
M = csvread(path, 1)
t = 0:1:(size(M,1)-1);
figure(fig); 

subplot(4,2,1)
xlabel('Epoch')
plot(t, M(:,1));
hold on
plot(t, M(:,8));
title('loss')
legend('train','val')


subplot(4,2,2)
xlabel('Epoch')
plot(t, M(:,2));
hold on
plot(t, M(:,9));
title('TPR')
legend('train','val')


subplot(4,2,3)
xlabel('Epoch')
plot(t, M(:,3));
hold on
plot(t, M(:,10));
title('TNR')
legend('train','val')


subplot(4,2,4)
xlabel('Epoch')
plot(t, M(:,4));
hold on
plot(t, M(:,11));
title('PPV')
legend('train','val')

subplot(4,2,5)
xlabel('Epoch')
plot(t, M(:,5));
hold on
plot(t, M(:,12));
title('NPV')
legend('train','val')



subplot(4,2,6)
xlabel('Epoch')
plot(t, M(:,6));
hold on
plot(t, M(:,13));
title('F1')
legend('train','val')

subplot(4,2,7)
xlabel('Epoch')
plot(t, M(:,7));
hold on
plot(t, M(:,14));
title('ACC')
legend('train','val')

end