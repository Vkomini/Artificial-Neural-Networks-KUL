clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'traingd'
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt 
%%%%%%%%%%%

%generation of examples and targets
x=0:0.2:3*pi; y=sin(x);
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%creation of networks
net1=feedforwardnet(5,'trainlm');
net2=feedforwardnet(5,'traingd');
net2.iw{1,1}=net1.iw{1,1}; %set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};

%training and simulation
net1.trainParam.epochs=1;  % set the number of epochs for the training
net2.trainParam.epochs=1;
net1=train(net1,p,t); % train the networks
net2=train(net2,p,t);
a11=sim(net1,p); a21=sim(net2,p);   % simulate the networks with the input vector p


net1.trainParam.epochs=14;
net2.trainParam.epochs=14;
net1=train(net1,p,t);
net2=train(net2,p,t);
a12=sim(net1,p); a22=sim(net2,p);

net1.trainParam.epochs=985;
net2.trainParam.epochs=985;
net1=train(net1,p,t);
net2=train(net2,p,t);
a13=sim(net1,p); a23=sim(net2,p);

%plots
figure
subplot(3,1,1);
plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a21),'g'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','trainlm','traingd',4);
%
subplot(3,1,2);
plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a22),'g');
title('15 epochs');
legend('target','trainlm','traingd',4);
%
subplot(3,1,3);
plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g');
title('1000 epochs');
legend('target','trainlm','traingd',4);
