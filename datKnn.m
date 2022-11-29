function [predictionsTest, K_nearest_indices,scores] = datKnn(dataTrain,dataTest,labelsTrain,K)

% Author: Serge Assaad, Sep 23, 2017
% Assumptions:
% - training and test data stored in matrices with samples as rows
% - labels_test is a column vector
% - L2 norm for distance
if(~iscolumn(labelsTrain))
    labelsTrain = labelsTrain';
end
N = size(dataTrain,1);
M = size(dataTest,1);

data_train_norm = diag(dataTrain*dataTrain');
data_test_norm = diag(dataTest*dataTest');

term1 = data_train_norm*ones(1,M);
term2 = data_test_norm*ones(1,N);
term3 = -2*dataTrain*dataTest';

distances_squared = term1+term2'+term3;

[~, sort_indices] = sort(distances_squared);
labels_train_matrix = labelsTrain*ones(1,M);
labels_train_sorted = labels_train_matrix(sort_indices);
K_nearest_indices = sort_indices(1:K,:);
K_nearest_labels = labels_train_sorted(1:K,:);
predictionsTest = mode(K_nearest_labels,1)';
scores = sum(K_nearest_labels)./K;
end