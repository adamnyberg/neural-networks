function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
cM = confusionmat(Lclass, Ltrue);
end

