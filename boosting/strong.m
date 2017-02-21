function [h] = strong(H, X)

    h = 0;

    for i = 1:size(H,2)
        % Weak classifiers features
        feature = H(2,i);
        threshold = H(3,i);
        polarity = H(4,i);
        alpha = H(5,i);
        x = X(feature);

        if x*polarity < threshold*polarity
           h = h + alpha;
        else
           h = h - alpha;
        end
    end

    h = sign(h);
end



