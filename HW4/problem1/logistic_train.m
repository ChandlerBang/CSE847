function [ weights ] = logistic_train(data, labels, epsilon, max_iter, lr ) 

[n, dim] = size(data); % Initialize weights with 0
weights = zeros(dim ,1) ;
acc=0.;
for i =1:max_iter
    % calculate the gradient 
    gradient = zeros(dim ,1) ; 
    for j = 1:n
        x_k = data(k,:);
        gradient = gradient + (sigmoid(x_k*weights) - labels(k))*(x_k)';
    end
    
    %gradient descent
    weights = weights - lr*gradient; 
    
    % calculate epsilon
    pred = sigmoid(data*weights);
    diff = pred' - labels;
    new_acc = sum(diff==0)/length(labels);
    if ( i == max_iter)  % ||  (abs((new_acc-acc)/new_acc) < epsilon)
        fprintf('Epsilon %f.\n', abs((new_acc-acc)/new_acc))
        fprintf('break at epoch %d', i);
        break
    end 
    acc = new_acc;
end
end