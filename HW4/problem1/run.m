function [prediction] = predict(data, weights) 
prediction = sigmoid(data∗weights); 
prediction ( prediction >=0.5) = 1; 
prediction ( prediction <0.5) = 0;
end
