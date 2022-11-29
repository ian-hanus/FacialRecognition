%% Initialize workspace
clc;
clf;
clear;

%% Load relevant data
% 400 face images, 10 for each subject in 112x92x400 
faces = load("faces.mat");
faces = faces.faces;

% 10 not face images in 10x10304
notfaces1 = load("not_faces.mat");
notfaces1 = notfaces1.not_faces;

% 400 not face images in 400x10304
notfaces2 = load("not_faces2.mat");
notfaces2 = notfaces2.not_faces;

%% Perform principal components analysis
% Rasterize the face matrix
for k = 1:length(faces)
    faceTest = faces(:, :, k);
    placeholder = faceTest(:);
    rasterize(:, k) = placeholder;
    clear faceTest placeholder;
end

for k = 1:length(faces)
    mu(k) = mean(rasterize(:, k));
    rasterize(:, k) = rasterize(:, k) - mu(k);
end

% % Perform SVD on the data
% [eigVecDat, L, eigVecCov] = svd(rasterize);
% Ldiag = diag(L);
% for k = 1:length(Ldiag)
%     eigValVec(k) = Ldiag(k).^2/ size(faces, 3);
% end
% eigVal = eigValVec .* eye(400);

% % Get the eigenvalues using eig
% [eigVec, eigVal] = eig(cov(rasterize));
% eigValVec = diag(eigVal);

% Use the Gram Matrix
gramMatrix = rasterize' * rasterize;
[eigVec, eigVal] = eig(gramMatrix);
eigValVec = diag(eigVal);

%% Scree Plot
plot(1:400, eigValVec);
title("Scree Plot")
xlabel("Number of Eigenvalues")
ylabel("Value")
axis([350 400 min(eigValVec) max(eigValVec)])

% saveas(gcf, "Scree.png")


%% Test elements
% testCase = eigVecCov(:, 1);
% testCase = reshape(testCase, 112, 92);
% imagesc(testCase)

%% Plot with correct dimensions
% allDims = faces(:, :, 1)*eigVec();
% % imagesc(allDims)






% testCase = faces(:, :, 1);
% testRast = testCase(:);


% for k = 1:40
%     person(k, :, :) = eigTotal((10*k - 9:10*k), :);
%     eigPerson(k, :) = mean(person(k,:, :), 2);
% end

% figure(2)
% hold on
% for k = 1:40
%     plot(1:92, eigPerson(k, :))
% end
% 
% % plot(1:92, specCov, 'b.')
% set(gca, 'xdir', 'reverse')
% axis([80 92 0 10000])



%% Project onto correct number of dimensions
% for k = 1:400
%     projectData(k, :, :) = squeeze(faces(:, :, k)) * squeeze(eigVec(k, :, :));
% end
% imagesc(squeeze(eigVec(3, :, :)))
% imagesc(squeeze(faces(:, :, 1)) * squeeze(eigVec(1, :, :)));
% project6 = projectData(:, :, 86:92);
% % 
% imagesc(squeeze(projectData(1, :, :)))