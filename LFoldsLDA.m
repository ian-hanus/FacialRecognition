%input projectdata truth_val optimalsets

function [peoplemean,pF_LDA,pD_LDA,max_AUC_LDA] = LFoldsLDA(projectdata,truth_val,optimalsets)


%% L number of k folds LDA - L = 10; k = 2
clear Keys



data_1 = projectdata(find(truth_val==1),:);
x=ceil(200/size(data_1,1));
faces_number_per_class = x*size(data_1,1) % number of faces per class - female=200; moustache=270; glasses=240
data_1 = repmat(data_1,x,1);
data_1 = data_1(1:faces_number_per_class,:);
true_1 = truth_val(find(truth_val==1));

for aa=1:faces_number_per_class
    Keys(:,aa)=[1:2];
end
Keys=Keys(:);

for Lfold=1:10
data_0 = projectdata(find(truth_val==0),:);
rand_num = size(data_0,1)
index = randperm(rand_num,faces_number_per_class);
data_0 = [data_0(index,:)];
true_0 = truth_val(find(truth_val==0));
data_rearranged = [data_1;data_0];
truth = [ones(1,faces_number_per_class)';zeros(1,faces_number_per_class)'];

numFolds=2;

kmax_1 = faces_number_per_class*2/10 - 1
for k = 0:kmax_1
    z(k+1) = 10*k + 1;
end
kmax_2 = faces_number_per_class*2/10
for k = 1:kmax_2
    q(k) = 10*k;
end


%%

for n=1:length(optimalsets)
projectdata_LDA=projectdata(:,optimalsets(n,:));
data=projectdata_LDA;
[pF_LDA,pD_LDA,decision_statistic] = LDAClassification(data,truth,Keys); %truth_female, truth_moustache, or truth_glasses
AUC_LDA(n) = trapz(pF_LDA,pD_LDA)
end



%%


AUC_LDA = AUC_LDA(:);
max_AUC_LDA = max(AUC_LDA);
max_comps_LDA = optimalsets(find(AUC_LDA==max(AUC_LDA)),:);

data_LDA_1 =data_rearranged(:,max_comps_LDA(1,:));
[pF_LDA,pD_LDA,decision_statistic_LDA] = LDAClassification(data_LDA_1,truth,Keys); %truth_female, truth_moustache, or truth_glasses


for k = 1:faces_number_per_class
    indexval = index(k);
    peoplematrix(indexval,Lfold) = decision_statistic_LDA(k);
end
clear indexval

rand_num_k = rand_num + 1;
for k = rand_num_k:400
    indexval = k;
    peoplematrix(indexval,Lfold) = decision_statistic_LDA(k);
end

end

peoplemean10folds = sum(peoplematrix,2) ./ sum(peoplematrix~=0,2); %averages the decision stats 400 faces for the 10 folds
n=1;
 for k = 1:40
     zvals = z(k);
     qvals = q(k);
     peoplemean(n) = mean(peoplemean10folds(zvals:qvals)); %averages the 10 image decision statistics
     n=n+1;
 end