im_sumit = imread('sumit.png');
im = ~im_sumit;
im = double(im);
im(im<1) = -1;

T = [];
for i=1:5:length(im)
    T = [T; reshape(im(:,i:i+4),1,35)];
end


caps = prprob;
caps(caps<1) = -1;
for i=1:26
    T = [T; reshape(reshape(caps(:,i),5,7)', 1, 35)];
end



net = newhop(T');

P = 5;
error = [];

%for P=1:30
    
    net = newhop(T(1:P,:)');

    NT = T(1:P,:);
    for i=1:P
        pos = randsample(35,3);
        for k=1:length(pos)
            NT(i,k) = -1*NT(i,k);
        end
    end

    Y = sim(net, P, [ ], NT');

    input = zeros(7,5*P);
    output = zeros(7,5*P);
    original = zeros(7,5*P);

    for i=1:P
        original(:,(i-1)*5 +1:(i-1)*5 +1 + 4) = reshape(T(i,:),7,5);
        input(:,(i-1)*5 +1:(i-1)*5 +1 + 4) = reshape(NT(i,:),7,5);
        output(:,(i-1)*5 +1:(i-1)*5 +1 + 4) = round(reshape(Y(:,i),7,5));
    end

    error = [error, sum(sum(abs(T(1:P,:)-round(Y'))))];

%end


plot(error)
grid on;
ylabel('Number of Wrong Pixels (Error)')
xlabel('Number of characters stored (P)')
title('Hopfield Network Error vs. Storage Capacity')
%saveas(gcf,'hopfield_storage_error.png');
figure, subimage(original)
figure, subimage(input)
figure, subimage(output)

