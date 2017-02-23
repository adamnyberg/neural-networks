function [Z] = mycorr(X, Y)
    Z = mycov(X, Y)/(sqrt(myvar(X)*myvar(Y)));
end