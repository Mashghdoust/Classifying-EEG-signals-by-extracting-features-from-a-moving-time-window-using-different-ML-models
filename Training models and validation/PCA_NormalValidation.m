%% loading data
%loading the features we have computed
load PCA_features;
%loading data sets 
load dataset_BCIcomp1;
%% defining the parameters
fs=128;
n_features=10; %number of PCA features
n_trials=size(x_train,3); %140
n_channels=size(x_train,2); %3
n=size(x_train,1); %1152
kernel_size=256; %as mentioned in the essay
s_point=n - kernel_size + 1; %897, which also describes number of different sets of features we can have
%% showing labels for a specific start point
%features when the start point is j and we choose the i'th feature
j=431;%start point
i=2;%number of the feature
sample1=PCA_features(i:n_features,:,j);
%finding the datas with 1 lable
leftindex1=find(y_train==1);
%finding the datas with 2 lable
rightindex1=find(y_train==2);
plot3(sample1(1,leftindex1),sample1(2,leftindex1),...
    sample1(3,leftindex1),'.','color','r');
xlabel('channel-1');
ylabel('channel-2');
zlabel('channel-3');
hold on
plot3(sample1(1,rightindex1),sample1(2,rightindex1),...
    sample1(3,rightindex1),'.','color','b');
legend('left-hand','right-hand');
grid on
grid minor
%% KNN
tp_accuracy=zeros(1,n_trials);
accuracy=zeros(1,s_point);
for k=[1 3 5 7]
    figure;
    for i=1:s_point
        %the dataset with the "i"th start point
        sample=PCA_features(:,:,i);
        % dividing data into test and split with the function we defined
        [sampleX_train,sampleX_test,sampley_train,sampley_test]=...
            train_test_split(0.7,sample,y_train);
        %training the model for KNN with k neighbours
        model=fitcknn(sampleX_train',sampley_train,'NumNeighbors',k);
        %testing the model
        op=predict(model,sampleX_test');
        %computing the accuracy
        accuracy(i)=sum(op==sampley_test)/length(sampley_test) *100;
    end
    best_point=find(max(accuracy)==accuracy);
    plot(accuracy);
    hold on
    stem(best_point,accuracy(best_point),'*','r');
    title(['KNN-k=',num2str(k)]);
    xlabel("start-point");
    ylabel("accuracy");
    grid on;
    grid minor;
    hold off;
end
%% SVM
%% linear
accuracy=zeros(1,s_point);
for i=1:s_point
    %the dataset with the "i"th start point
    sample=PCA_features(:,:,i);
    %dividing data into test and split with the function we defined
    [sampleX_train,sampleX_test,sampley_train,sampley_test]=...
        train_test_split(0.7,sample,y_train);
    %training the model with linear kernel
    model=fitcsvm(sampleX_train',sampley_train,'Standardize',1,...
        'KernelFunction','linear');
    %testing the model
    op=predict(model,sampleX_test');
    %computing the accuracy
    accuracy(i)=sum(op==sampley_test)/length(sampley_test) *100;
end
best_point=find(max(accuracy)==accuracy);
figure;
plot(accuracy);
hold on
stem(best_point,accuracy(best_point),'*','r');
title('SVM-linear-kernel');
xlabel("start-point");
ylabel("accuracy");
grid on;
grid minor;
hold off;
%% gaussian
accuracy=zeros(1,s_point);
for i=1:s_point
    %the dataset with the "i"th start point
    sample=PCA_features(:,:,i);
    %dividing data into test and split with the function we defined
    [sampleX_train,sampleX_test,sampley_train,sampley_test]=...
        train_test_split(0.7,sample,y_train);
    %training the model with gaussian kernel
    model=fitcsvm(sampleX_train',sampley_train,'Standardize',1,...
        'KernelFunction','RBF','KernelScale','auto');
    %testing the model
    op=predict(model,sampleX_test');
    %computing the accuracy
    accuracy(i)=sum(op==sampley_test)/length(sampley_test) *100;
end
best_point=find(max(accuracy)==accuracy);
figure;
plot(accuracy);
hold on
stem(best_point,accuracy(best_point),'*','r');
title('SVM-gaussian-kernel');
xlabel("start-point");
ylabel("accuracy");
grid on;
grid minor;
hold off;

%% LDA
accuracy=zeros(1,s_point);
for i=1:s_point
    %the dataset with the "i"th start point
    sample=PCA_features(:,:,i);
    %dividing data into test and split with the function we defined
    [sampleX_train,sampleX_test,sampley_train,sampley_test]=...
        train_test_split(0.7,sample,y_train);
    %training the model using LDA
    model=fitcdiscr(sampleX_train',sampley_train);
    %testing the model
    op=predict(model,sampleX_test');
    %computing the accuracy
    accuracy(i)=sum(op==sampley_test)/length(sampley_test) *100;
end
best_point=find(max(accuracy)==accuracy);
figure;
plot(accuracy);
hold on
stem(best_point,accuracy(best_point),'*','r');
title('LDA');
xlabel("start-point");
ylabel("accuracy");
grid on;
grid minor;
hold off;
%% Naive bayes
accuracy=zeros(1,s_point);
for i=1:s_point
    %the dataset with the "i"th start point
    sample=PCA_features(:,:,i);
    %normalizing data
    sample=[sample(1,:)/max(sample(1,:)) ; sample(2,:)/max(sample(2,:)) ;...
        sample(3,:)/max(sample(3,:))];
    %dividing data intos test and split with the function we defined
    [sampleX_train,sampleX_test,sampley_train,sampley_test]=...
        train_test_split(0.7,sample,y_train);
    %training the model using naive bayes
    model=fitcnb(sampleX_train',sampley_train);
    %testing the model
    op=predict(model,sampleX_test');
    %computing the accuracy
    accuracy(i)=sum(op==sampley_test)/length(sampley_test) *100;
end
best_point=find(max(accuracy)==accuracy);
figure;
plot(accuracy);
hold on
stem(best_point,accuracy(best_point),'*','r');
title('naive-bayes');
xlabel("start-point");
ylabel("accuracy");
grid on;
grid minor;
hold off;
