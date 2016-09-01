function regress

    load('Data_Problem1_regression.mat');

    %r0607761
    d1=7; d2=7; d3=6; d4=6; d5=1;
    Tnew = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5)/(d1 + d2 + d3 + d4 + d5);

    splitPoint = 0.8*length(Tnew);
    trainset = Tnew(1:splitPoint);
    testset = Tnew(splitPoint+1:end);

    Xtrain = [X1(1:splitPoint) X2(1:splitPoint)];
    Xtest = [X1(splitPoint+1:end) X2(splitPoint+1:end)]; size(Xtrain)
    
    figure, plotSurfaceQ1(X1(1:splitPoint), X2(1:splitPoint), trainset,'Training Set')
    figure, plotSurfaceQ1(X1(splitPoint+1:end), X2(splitPoint+1:end), testset,'Test Set')

    algo = {'traingd', 'trainlm', 'trainbr'};
    %algo = {'trainlm'};
    hiddenUnits = [10, 100];
    EpochsArr = [1000];

    for i=1:length(algo)
        for k=1:length(hiddenUnits)
            cumEpoch = 0;
            for ep=1:length(EpochsArr)
                net=feedforwardnet(hiddenUnits(k), algo{i});
                net.trainParam.epochs=EpochsArr(ep) - cumEpoch;
                net=train(net, Xtrain', trainset');
                op = sim(net,Xtest');
                error = mse(testset, op');
                figure
                title_str = sprintf('%s, %d Hidden units, %d Epochs\nmse = %f', algo{i}, hiddenUnits(k), EpochsArr(ep), error);
                plotSurfaceQ1(X1(splitPoint+1:end), X2(splitPoint+1:end), op',title_str);
                title_str = sprintf('%s, %d Hidden units, %d Epochs', algo{i}, hiddenUnits(k), EpochsArr(ep));
                title_str = strcat(strip(title_str),'.png');
                saveas(gcf, strcat('Images/regression/new/', title_str));
                %set(gcf, 'visible', 'off');
                cumEpoch = cumEpoch + EpochsArr(ep);
            end
        end
    end

end

function h=plotSurfaceQ1(X1,X2,T,titleStr)
    XVEC = X1;
    YVEC = X2;
    ZVEC = T;
    size(X1)
    size(X2)
    size(T)
    F=scatteredInterpolant(XVEC,YVEC,ZVEC);
    [Xq,Yq]=meshgrid(min(XVEC):0.01:max(XVEC),min(YVEC):0.01:max(YVEC));
    Vq = F(Xq,Yq);
    h=surfc(Xq,Yq,Vq);
    xlabel('X_1');
    ylabel('X_2');
    zlabel('T');
    title(titleStr);
end

function title_str = strip(title_str)
    title_str = strrep(title_str, '=', '');
    title_str = strrep(title_str, '\n', '');
    title_str = strrep(title_str, ' ', '');
    title_str = strrep(title_str, ',', '');
    title_str = strrep(title_str, '(', '');
    title_str = strrep(title_str, ')', '');
end
