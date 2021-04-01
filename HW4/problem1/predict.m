% get the prediction
function pred = predict(data, weights)
    N = size(data,1);
    pred = zeros(N,1);
    for i=1:N
        value = sigmoid(data(i,:) * weights);
        if value >= 0.5
            pred(i) = 1;
        else
            pred(i) = 0;
        end
    end
end

