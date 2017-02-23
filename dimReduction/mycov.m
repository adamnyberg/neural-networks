function [Z] = mycov(X, Y)
    n = size(X, 2);
    res = 0;
    
    Xmean = sum(X)/n;
    Ymean = sum(Y)/n;
    
    for i = 1:n
        res = res + ((X(i) - Xmean) * (Y(i) - Ymean)');
    end
    Z = res/(n-1);
end
