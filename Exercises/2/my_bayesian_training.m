function my_bayesian_training(algo)

%generation of examples and targets
x=0:0.2:3*pi; 

y=sin(x); %label = {'sin(x)'};
label = {'sin(x) + Noise'};

%y=sin(x.^2); label = {'y = sin(x.^2)'};
%y=log(x).*sin(x).*sin(x.^2); label = {'y = log(x).*sin(x).*sin(x.^2)'};
y = awgn(y,10,'measured'); 
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%algo = algo{1};
hiddenUnits = [5, 15, 50];

for k=1:length(hiddenUnits)
    
    %creation of networks
    net1=feedforwardnet(hiddenUnits(k), 'trainbr');
    net2=feedforwardnet(hiddenUnits(k), algo);
    net2.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
    net2.lw{2,1}=net1.lw{2,1};
    net2.b{1}=net1.b{1};
    net2.b{2}=net1.b{2};


    EpochsArr = [1, 15, 1000];
    cumEpoch = 0;
    %training and simulation
    for i=1:length(EpochsArr)
        net1.trainParam.epochs=EpochsArr(i) - cumEpoch; 
        net2.trainParam.epochs=EpochsArr(i) - cumEpoch;
        
        net1=train(net1,p,t);
        net2=train(net2,p,t);
        opGd=sim(net1,p); opAlgo=sim(net2,p);  % simulate the networks with the input vector p
        
        
        
        figure
        plot(x,y,'bx',x,cell2mat(opGd),'r',x,cell2mat(opAlgo),'g');
        grid on;
        title_str = sprintf('With Noise, %d Epochs, %d Hidden units (%s)', EpochsArr(i), hiddenUnits(k), algo);
        title(title_str);
        legend('target', 'trainbr', algo, 4);
        ylabel(label{1});
        xlabel('x');
        title_str = strcat('Images/sine/', title_str);
        
        set(gcf,'visible','off');
        
        saveas(gcf,strcat(strip(title_str),'.png'));
        %{
        title_str = sprintf('With Noise, %d Epochs, %d Hidden units (%s)', EpochsArr(i), hiddenUnits(k), 'trainbr');
        postregm(cell2mat(opGd),y, title_str);
        grid on;
        title_str = strcat('Images/sine/R_', title_str);
        saveas(gcf,strcat(strip(title_str),'.png'));
        
        title_str = sprintf('With Noise, %d Epochs, %d Hidden units (%s)', EpochsArr(i), hiddenUnits(k), algo);
        postregm(cell2mat(opAlgo),y, title_str);
        grid on;
        title_str = strcat('Images/sine/R_', title_str);
        saveas(gcf,strcat(strip(title_str),'.png'));
        %}
        cumEpoch = cumEpoch + EpochsArr(i);
        
    end
    
end
close all
end


function title_str = strip(title_str)
    title_str = strrep(title_str, ' ', '');
    title_str = strrep(title_str, ',', '');
    title_str = strrep(title_str, '(', '');
    title_str = strrep(title_str, ')', '');
end