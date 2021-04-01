load('/home/jinwei/Downloads/cse847/CSE847/data/alzheimers/ad_data.mat')
addpath('/home/jinwei/Downloads/cse847/SLEP', '/home/jinwei/Downloads/cse847/SLEP/SLEP/opts')
arr_lambda = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
auc_vec = zeros(1, length(arr_lambda)); 
nonzero = zeros(1, length(arr_lambda)); 

for i = 1:length(arr_lambda)
    lambda = arr_lambda(i);
    opts.rFlag = 1;
    opts.tol = 1e-6;
    opts.tFlag = 4;
    opts.maxIter = 5000;
    [w,c] = LogisticR(X_train, y_train, lambda, opts); 
    pred = X_test * w + c * ones(length(X_test(:,1)),1);
    [a,b,cc, AUC] = perfcurve(y_test, pred , 1);
    auc_vec(i) =AUC;
    nonzero(i) = nnz(w);
end

