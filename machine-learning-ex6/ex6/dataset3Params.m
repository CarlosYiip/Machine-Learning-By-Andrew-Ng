function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 2;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

prev_err = 1000000;
for c = [0.01 0.03 0.1 0.3 1 3 10 30]
	for s = [0.01 0.03 0.1 0.3 1 3 10 30]
		model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
		prediction = svmPredict(model, Xval);
		err = mean(double(prediction ~= yval));
		if (err < prev_err)
			C = c;
			sigma = s;
			prev_err = err;
		endif
	end;
end;
% =========================================================================
end
