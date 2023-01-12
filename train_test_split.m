%% dividing datas into train and test

function [X_train,X_test,y_train,y_test] = train_test_split(percent,X,y)

num=round(percent*size(X,2));

X_train=X(:,1:num);
X_test=X(:,num+1:end);
y_train=y(1:num);
y_test=y(num+1:end);
end
