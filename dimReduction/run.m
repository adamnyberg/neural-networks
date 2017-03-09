load countrydata;

N = size(countrydata, 2);
M = size(countrydata, 1);

norm_data = zeros(M, N);
for i = 1:M
    norm_data(i,:) = (countrydata(i,:) - min(countrydata(i,:))) ./ (max(countrydata(i,:))-min(countrydata(i,:)));
end

cov_matrix = zeros(M, M);
for i = 1:M
    for j = 1:M
        cov_matrix(i, j) = mycov(countrydata(i,:), countrydata(j,:));
    end
end
figure(1);
imagesc(cov_matrix);

corr_matrix = zeros(M, M);
for i = 1:M
    for j = 1:M
        corr_matrix(i, j) = mycorr(countrydata(i,:), countrydata(j,:));
    end
end
figure(2);
imagesc(corr_matrix);


%%
% PCA

cov_norm = zeros(M, M);
for i = 1:M
    for j = 1:M
        cov_norm(i, j) = mycov(norm_data(i,:), norm_data(j,:));
    end
end

[W, V] = sorteig(cov_norm);

pc1_percent = V(1)/sum(V);
pc2_percent = V(2)/sum(V);

percent = zeros(length(V));
for i = 1:length(V)
    percent(i) = V(i)/sum(V);
end
figure(4);
plot(percent);

figure(5); title 'PCA'; hold on;
markers = {'go', 'bo','mo'};
for class = 0:length(unique(countryclass))-1
    X = norm_data(:,countryclass == class);
    PC1 = W(:,1)'*X;
    PC2 = W(:,2)'*X;
    scatter(PC1, PC2, markers{class+1});
end;

georgia_PC1 = W(:,1)'*norm_data(:,41);
georgia_PC2 = W(:,2)'*norm_data(:,41);
plot(georgia_PC1, georgia_PC2, 'r+');



xlabel 'PC1'; ylabel 'PC2'; legend 'developing' 'inbetween' 'industrialized' 'Georgia';
hold off;


%%

class_data = norm_data(:, (mod(countryclass, 2) == 0));

class_cov = zeros(M, M);
for i = 1:M
    for j = 1:M
        class_cov(i, j) = mycov(class_data(i,:), class_data(j,:));
    end
end

w = class_cov\(mean(norm_data(:,countryclass == 0), 2)-mean(norm_data(:,countryclass == 2), 2));

FLD1 = w'*norm_data(:,countryclass==0);
FLD2 = w'*norm_data(:,countryclass==2);

% plotting 
figure(6); clf; hold on;
scatter(FLD1,zeros(size(FLD1)), 'go');
scatter(FLD2,zeros(size(FLD2)), 'mo');
xlabel 'FLD1'; ylabel 'FLD2'; legend 'developing' 'industrialized';
hold off;







