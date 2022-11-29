%% Project 3: KNN, LR, and LDA

%% Initialize workspace
clear; clc; clf;

%% Load relevant data
% 400 face images, 10 for each subject in 112x92x400
faces = load('faces.mat');
faces = faces.faces;

% 10 not face images in 10x10304
notfaces1 = load('not_faces.mat');
notfaces1 = notfaces1.not_faces;

% 400 not face images in 400x10304
notfaces2 = load('not_faces2.mat');
notfaces2 = notfaces2.not_faces;

%% Rasterize the data
for k = 1:length(faces)
    faceTest = faces(:, :, k);
    placeholder = faceTest(:);
    rasterize(:, k) = placeholder;
    clear faceTest placeholder;
end

%% Mean center the rasterized data
for k = 1:length(faces)
    mu(k) = mean(rasterize(:, k));
    rasterize(:, k) = rasterize(:, k) - mu(k);
end

%% Find scree plot using eigendecomposition
eigData = load('eigDecomp.mat');
eigVec1 = eigData.eigVec1;
eigVal1 = eigData.eigVal1;

totalVals1 = diag(eigVal1);
[sortedVals1, sortedVecIndices1] = sort(totalVals1, 'descend');
sortedVecs1 = eigVec1(:, sortedVecIndices1);

% testcase= faces(:,:,391);
% %testcase=reshape(testcase,92,112);
% imagesc(testcase)
% colormap('gray')
% axis('image')

%% Project Data
projectdata=rasterize'*sortedVecs1;

%% Create truth vector for KNN
% A=ones(10,1);
% B=1:40;
% mat=A*B;
% truth_KNN=mat(:);
% numFolds=10;
% for aa=1:40
%     Keys(:,aa)=[1:10];
% end
% Keys=Keys(:);

%% Create sets of principal components for classification
sets_1=nchoosek(1:8,1);
sets_2=nchoosek(1:8,2);
sets_3=nchoosek(1:8,3);
sets_4=nchoosek(1:8,4);
sets_5=nchoosek(1:8,5);
sets_6=nchoosek(1:8,6);
sets_7=nchoosek(1:8,7);
sets_8=nchoosek(1:8,8);

%% KNN
%for n=1:length(sets_4)
projectdata_KNN=projectdata(:,[1 2 3 4 5 6 8]);

%% Create TSNE Plot
% rng default % for reproducibility
% Y = tsne(projectdata_KNN);
% gscatter(Y(:,1),Y(:,2),species)

%% Run KNN on original data set
% K=3;
% data=rasterize';
% 
% %  for thisFold=1:numFolds
% testData=[data(Keys==thisFold,:)];
% trainData=[data(Keys~=thisFold,:)];
% truth_thisFold=truth_KNN(Keys~=thisFold);
% [predictionsTest(Keys==thisFold), k_Indices(:,Keys==thisFold), scores(Keys==thisFold)]=datKnn(trainData,testData,truth_thisFold,K);
% thisFold=thisFold+1;
% end
% 
% for a=1:400
%     if predictionsTest(a)==truth_KNN(a)
%         probvector(a)=1;
%     else
%         probvector(a)=0;
%     end
% end
% 
% prob_correct=sum(probvector)./400;
    
%% Cross-Validation KNN: 10-fold Cross Validation with PCA Data
%for K=1:15
% K=3
% data=projectdata_KNN;
% 
% for thisFold=1:numFolds
%     testData=[data(Keys==thisFold,:)];
%     trainData=[data(Keys~=thisFold,:)];
%     truth_thisFold=truth_KNN(Keys~=thisFold);
%     [predictionsTest(Keys==thisFold), k_Indices(:,Keys==thisFold), scores(Keys==thisFold)]=datKnn(trainData,testData,truth_thisFold,K);
%     thisFold=thisFold+1;
% end
% 
% for a=1:400
%     if predictionsTest(a)==truth_KNN(a)
%         probvector(a)=1;
%     else
%         probvector(a)=0;
%     end
% end
% 
% prob_correct=sum(probvector)./400;
% %clear predictionsTest probvector k_Indices scores
% %end
% 
% % Calculate probability correct by subject
% for w=1:391
%     prob_subject(w)=sum(probvector(w:w+9));
% end
% prob_subject=prob_subject(1:10:400)./10;

% figure(1)
% plot(1:15,prob_correct,'k-o')
% title('Probability Correct with Varying K','interpreter','latex','fontsize',12);
% xlabel('K','interpreter','latex','fontsize',12);
% ylabel('Probability Correct','interpreter','latex','fontsize',12);

%find(prob_correct==max(prob_correct))
%max(prob_correct)

%% Create truth vectors for binary classification
% moustache vs. no moustache
for k = 0:39
    z(k+1) = 10*k + 1
end
for k = 1:40
    q(k) = 10*k
end
 for k = [1:6,8:10,12:13,15,18:24,27,29:36,38:40]
     zvals = z(k)
     qvals = q(k)
     truth_moustache(zvals:qvals) = 0 % no moustache
 end
  for k = [7,11,14,16:17,25:26,28,37]
     zvals = z(k)
     qvals = q(k)
     truth_moustache(zvals:qvals) = 1 % moustache
  end
  
% male v. female
for k = [1:7,9,11:31,33:34,36:40]
    zvals = z(k)
    qvals = q(k)
    truth_female(zvals:qvals) = 0 % male
end
for k = [8,10,32,35]
    zvals = z(k)
    qvals = q(k)
    truth_female(zvals:qvals) = 1 % female
end

% Glasses v. no glasses
for k = [1,3,5,7:12,15:16,18,21:26,29:30,32:33,35:36,38:40]
    zvals = z(k)
    qvals = q(k)
    truth_glasses(zvals:qvals) = 0 % no glasses
end
for k = [2,4,6,13:14,17,19:20,27:28,31,37]
    zvals = z(k)
    qvals = q(k)
    truth_glasses(zvals:qvals) = 1 % glasses
end

%% LR
% LR:male v. female
clear Keys
[peoplemean_gender,pF_LR_gender,pD_LR_gender,max_AUC_gender] = LFoldsLR(projectdata,truth_female,sets_6);
peoplemean_gender=peoplemean_gender(:)

% LR: moustache v. no moustache
[peoplemean_moustache,pF_LR_moustache,pD_LR_moustache,max_AUC_moustache] = LFoldsLR(projectdata,truth_moustache,sets_7);
peoplemean_moustache=peoplemean_moustache(:)

% LR: glasses v. no glasses
[peoplemean_glasses,pF_LR_glasses,pD_LR_glasses,max_AUC_glasses] = LFoldsLR(projectdata,truth_glasses,sets_4);
peoplemean_glasses=peoplemean_glasses(:)
DECSTAT_LR = [peoplemean_gender,peoplemean_moustache,peoplemean_glasses]
Max_AUC_vals_LR = [max_AUC_gender,max_AUC_moustache,max_AUC_glasses]

%% LDA
% LDA for All Characteristics
% [peoplemean_LDA_gender,pF_LDA_gender,pD_LDA_gender,max_AUC_LDA_gender] = LFoldsLDA(projectdata,truth_female,sets_4);
% [peoplemean_LDA_moustache,pF_LDA_moustache,pD_LDA_moustache,max_AUC_LDA_moustache] = LFoldsLDA(projectdata,truth_moustache,sets_7);
% [peoplemean_LDA_glasses,pF_LDA_glasses,pD_LDA_glasses,max_AUC_LDA_glasses] = LFoldsLDA(projectdata,truth_glasses,sets_5);
% Max_AUC_vals_LDA = [max_AUC_LDA_gender,max_AUC_LDA_moustache,max_AUC_LDA_glasses];

%% Plot ROC Curves Together for All Variables
figure(1);clf;
% plot(pF_LDA_gender,pD_LDA_gender,'k-')
hold on
plot(pF_LR_gender,pD_LR_gender,'b-')
xlabel('p$_{FA}$','interpreter','latex','fontsize',12);
ylabel('p$_{D}$','interpreter','latex','fontsize',12);
title('ROC Curve with 10-Run 2-Fold Cross Validation: Male vs. Female','interpreter','latex','fontsize',12);
% legend('LDA','LR')
set(gca,'FontName','Times New Roman');

figure(2);clf;
% plot(pF_LDA_moustache,pD_LDA_moustache,'k-')
hold on
plot(pF_LR_moustache,pD_LR_moustache,'b-')
xlabel('p$_{FA}$','interpreter','latex','fontsize',12);
ylabel('p$_{D}$','interpreter','latex','fontsize',12);
title('ROC Curve with 10-Run 2-Fold Cross Validation: Moustache?','interpreter','latex','fontsize',12);
% legend('LDA','LR')
set(gca,'FontName','Times New Roman');

figure(3);clf;
% plot(pF_LDA_glasses,pD_LDA_glasses,'k-')
hold on
plot(pF_LR_glasses,pD_LR_glasses,'b-')
xlabel('p$_{FA}$','interpreter','latex','fontsize',12);
ylabel('p$_{D}$','interpreter','latex','fontsize',12);
title('ROC Curve with 10-Run 2-Fold Cross Validation: Glasses?','interpreter','latex','fontsize',12);
% legend('LDA','LR')
set(gca,'FontName','Times New Roman');