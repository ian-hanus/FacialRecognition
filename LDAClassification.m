function[pF_LDA,pD_LDA,decision_statistic] = LDAClassification(data,labels,Keys)

% % % % % % % %   Y0 = data(find(labels==0),:);
% % % % % % % %   Keys_HO = Keys(find(labels==0),:);
% % % % % % % %   Y1 = data(find(labels==1),:);
% % % % % % % %   Keys_H1 = Keys(find(labels==1),:);
% % % % % % % %   L0 = labels(find(labels==0));
% % % % % % % %   L1 = labels(find(labels==1));
% % % % % % % %   data = [Y0;Y1];
% % % % % % % %   L0 = L0';
% % % % % % % %   L1 = L1';
% % % % % % % %   labels = [L0;L1];
% % % % % % % %   truth = labels;
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


numFolds=2;
% Keys_HO=(rem(randperm(length(Y0))-1, numFolds)+1);
% Keys_H1=(rem(randperm(length(Y1))-1, numFolds)+1);
% Keys=[Keys_HO, Keys_H1]';

for thisFold=1:numFolds
    testData=[Y0(Keys_HO==thisFold,:);Y1(Keys_H1==thisFold,:)];
    %testData=[Y0(Keys==thisFold,:);Y1(Keys==thisFold,:)];
    traindata_Y1=[Y1(Keys_H1~=thisFold,:)];
    traindata_Y0=[Y0(Keys_HO~=thisFold,:)];
    %traindata_Y1=[Y1(Keys~=thisFold,:)]
    %traindata_Y0=[Y0(Keys~=thisFold,:)]
    traindata = [traindata_Y0;traindata_Y1];
    truth_thisFold=truth(Keys~=thisFold);
pi0Hat = size(traindata_Y0,1)./size(traindata,1);
pi1Hat = size(traindata_Y1,1)./size(traindata,1);

mu0Hat = mean(traindata_Y0)';
mu1Hat = mean(traindata_Y1)';

sigmaHat = cov([traindata_Y0(:,1)-mu0Hat(1)';traindata_Y1(:,1)-mu1Hat(1)']);
XTest = testData;
decision_statistic(Keys==thisFold) = XTest*sigmaHat*(mu1Hat-mu0Hat)+log(pi1Hat)-log(pi0Hat)-0.5*mu1Hat'*inv(sigmaHat)*mu1Hat+0.5*mu0Hat'*inv(sigmaHat)*mu0Hat;

    thisFold=thisFold+1;
end

[pF_LDA,pD_LDA] = datDecStat2ROC(decision_statistic,truth);


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