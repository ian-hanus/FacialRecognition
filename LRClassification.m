function[pF_LR,pD_LR,decision_statistic] = LRClassification(data,labels,Keys)
% coeffs = load('coeffs_A.mat')
% coeffs = coeffs.coeffs
% coeffs_B = load('coeffs_B.mat')
% coeffs_B = coeffs_B.coeffs_B

%% Use LR to Classify Test Data
% 
  Y0 = data(find(labels==0),:);
  Keys_HO = Keys(find(labels==0),:);
  Y1 = data(find(labels==1),:);
  Keys_H1 = Keys(find(labels==1),:);
  L0 = labels(find(labels==0));
  L1 = labels(find(labels==1));
  data = [Y0;Y1];
  L0 = L0(:);
  L1 = L1(:);
  labels = [L0;L1];
  truth = labels;
% 
% % Decision surface parameters
% numx1Test = 101;
% x1TestVec = linspace(min(min(Y0)),max(max(Y1)),numx1Test);
% labels = labels'
% % Logistic Regression
% coefs = glmfit(data,labels,'binomial','link','logit');
% 
% % Sigmoid Function
 sigmoid = @(z) 1./(1+exp(-z));

% Plot decision surface
% numx1Test = length(data);
% numx2Test = length(data);
% numx3Test = length(data);
% numx4Test = length(data);
% numx5Test = length(data);
% numx6Test = length(data);
% numx7Test = length(data);
% x1TestVec = linspace(min(Y0(:,1)),max(Y1(:,1)),numx1Test);
% x2TestVec = linspace(min(Y0(:,2)),max(Y1(:,2)),numx2Test);
% x3TestVec = linspace(min(Y0(:,3)),max(Y1(:,3)),numx3Test);
% x4TestVec = linspace(min(Y0(:,4)),max(Y1(:,4)),numx4Test);
% x5TestVec = linspace(min(Y0(:,5)),max(Y1(:,5)),numx5Test);
% x6TestVec = linspace(min(Y0(:,6)),max(Y1(:,6)),numx6Test);
% x7TestVec = linspace(min(Y0(:,7)),max(Y1(:,7)),numx7Test);
% [x1TestMesh,x2TestMesh,x3TestMesh,x4TestMesh,x5TestMesh,x6TestMesh,x7TestMesh] = ndgrid(x1TestVec,x2TestVec,x3TestVec,x4TestVec,x5TestVec,x6TestVec,x7TestVec);
% XTest = [ones(numx1Test*numx2Test*numx3Test*numx4Test*numx5Test*numx6Test*numx7Test,1)  x1TestMesh(:)  x2TestMesh(:) x3TestMesh(:) x4TestMesh(:) x5TestMesh(:) x6TestMesh(:) x7TestMesh(:)];
% %decSurf = sigmoid(XTest*coefs);



numFolds=2;
% Keys_HO=(rem(randperm(length(Y0))-1, numFolds)+1);
% Keys_H1=(rem(randperm(length(Y1))-1, numFolds)+1);
% Keys=[Keys_HO, Keys_H1]';

for thisFold=1:numFolds
    testData=[Y0(Keys_HO==thisFold,:);Y1(Keys_H1==thisFold,:)];
    %testData = data(Keys==thisFold,:)
    trainData=[Y0(Keys_HO~=thisFold,:); Y1(Keys_H1~=thisFold,:)];
    %trainData = data(Keys~=thisFold,:)
    truth_thisFold=truth(Keys~=thisFold);
    coeffs = glmfit(trainData,truth_thisFold,'binomial','link','logit');
    decision_statistic(Keys==thisFold)=sigmoid([ones(length(testData),1) testData]*coeffs);
    thisFold=thisFold+1;
end
[pF_LR,pD_LR] = datDecStat2ROC(decision_statistic,truth);


% Plot decision surface
% numxTest = 500;
% numyTest = 500;
% xTestVec = linspace(0,1,numxTest);
% yTestVec = linspace(0,1,numyTest);
% 
% [meshDataX, meshDataY]=meshgrid(xTestVec,yTestVec);
% TestData=[ones(numxTest*numyTest,1) meshDataX(:) meshDataY(:)];
% 
% decSurf = sigmoid(TestData*coefs);
% decSurf = reshape(decSurf,numxTest,numyTest);
% 
% figure
% imagesc(xTestVec,yTestVec,decSurf,[0 1])
% set(gca,'YDir','normal')
% hold on
% plot(Y0(:,1),Y0(:,2),'s','MarkerEdgeColor',[1 0.5 0],'MarkerFaceColor','y','LineWidth',1)
% hold on
% plot(Y1(:,1),Y1(:,2),'^','MarkerEdgeColor','b','MarkerFaceColor','c','LineWidth',1)
% colorbar
% legend('Type_A','Type_B')
% title({'Logistic Regression';'Decision Surface for p(H_1)'})
end