function [acc] = accuracy(classifiers, X, Y)

    N = size(Y, 2);
    correct = 0;
    h1 = 0;

    for i = 1:N

        h1 = strong(classifiers, X(:,i));

        if h1 == Y(1,i)
            correct = correct +1;   
        end

    end

    acc = correct/N;
end
