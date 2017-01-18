function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
%CALCCONFUSIONMATRIX Summary of this function goes here
%   Detailed explanation goes here

classes = unique(Ltrue);
numClasses = length(classes);
cM = zeros(numClasses);


end

