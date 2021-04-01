% load data
x = load('/home/jinwei/Downloads/cse847/CSE847/data/spam_email/data.txt');
y = load('/home/jinwei/Downloads/cse847/CSE847/data/spam_email/labels.txt');
n_train = [200, 500, 800, 1000, 2000]; 

[N, tmp] = size(x); 

% add one column vector
x = [x ones(N, 1)];

x_test = x(2001: N, :); 
y_test = y(2001: N);

for i = 1: length(n_train)
    fprintf('===n_train=%d===\n', n_train(i))
    x_train = x(1:n_train(i), :);
    y_train = y(1:n_train(i));
     
    weights = logistic_train(x_train, y_train, 1e-5, 5000, 0.005); 

    %% get training accuracy
    % x_test = x_train; y_test = y_train;   
    pred = predict(x_test, weights);
    diff = pred - y_test;
    acc = sum(diff==0)/length(y_test);
    fprintf('acc: %f\n', acc)
end
