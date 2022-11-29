function [covMat, eigVec, eigVal, eigTotal] = PCA(data)

%% Calculate covariance matrix
covMat = cov(data);

%% Compute spectrum of the covariance matrix (eigenvalues)
[eigVec, eigVal] = eig(covMat);
eigTotal = eig(covMat);

%% Examine eigenvalues and determine if dimension reduction is appropriate
% specCount = 1;
% dimRed = 0;
% for i = 1:length(specCov)
%    for j = 1:length(specCov)
%       if(i ~= j && abs(specCov(i) - specCov(j)) < 0.2)
%           dimRed(specCount) = i;
%           specCount = specCount + 1;
%       end
%    end
% end
% dimRed = unique(dimRed);

%% Project the data onto the span of the appropriate collection of principal components