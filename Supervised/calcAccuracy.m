function [ acc ] = calcAccuracy( cM )
acc = sum(diag(cM)) / sum(cM(:)); % Replace with your own code
end

