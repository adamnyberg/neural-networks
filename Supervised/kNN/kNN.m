function [ labelsOut ] = kNN(test, k, train, train_classes)

preds = [];

dist_matrix = pdist2(test',train');

for row = 1:length(test)
    
    current_dist = dist_matrix(row,:);
    
    [throwaway, SortIndex] = sort(current_dist);
    Ysorted = train_classes(SortIndex);
    
    k_nearest = Ysorted(1:k);
    prediction = round(sum(k_nearest)/length(k_nearest));
    preds(row) = prediction;
    
end

labelsOut  = preds';

end

