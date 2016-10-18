clear; close all; clc;

printf("*** Handwriting Classification Demo ***\n");

% This file loads 'X' and 'y' values
load('data/ex4data1.mat');
% Shuffle up the rows of the 'X' and 'y' input data
perms = randperm(size(X, 1));
shuffled_X = X(perms, :);
shuffled_y = y(perms);
% Split 'X' and 'y' into "train" and "test" sets
m = size(shuffled_X, 1);
num_train = round(m * .7);
X_train = shuffled_X(1:num_train, :);
X_test = shuffled_X(num_train+1:end, :);
% The 'y' input data has values, 1-10, with 10 representing
% zero. We want to convert that into a matrix with column for each 
% number as a feature, e.g. number '2' equals [0, 1, 0, ...]
y_multiclass = [shuffled_y==1 shuffled_y==2 shuffled_y==3 shuffled_y==4 shuffled_y==5 shuffled_y==6 shuffled_y==7 shuffled_y==8 shuffled_y==9 shuffled_y==10];
y_train = y_multiclass(1:num_train, :);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
display_data(X(sel, :));
printf("Hit any key to continue\n");
pause;

[theta1, theta2] = nn_train(X_train, y_train, 0.001, max_iterations = 1000);

pred_values = nn_predict(X_test, theta1, theta2);
[probability, pred_number] = max(pred_values, [], 2);
pred_errors = pred_number != shuffled_y(num_train+1:end, :);
num_test_observations = size(X_test, 1);
num_test_errors = sum(pred_errors);
num_test_hits = num_test_observations - num_test_errors;
accuracy = 100.0 * num_test_hits / num_test_observations;
printf("\n\n*** The model was %5.2f%% accurate against the test set. ***\n", accuracy);
printf("Hit any key to continue\n");

% Find all the errors and display them 
y_shuffled_test = shuffled_y(num_train+1:end, :);
error_rows = find(pred_errors);
X_errors = X_test(error_rows, :);
y_errors = pred_number(error_rows, :);
y_actual = y_shuffled_test(error_rows);
prob_errors = probability(error_rows, :);

printf("Walking through each error.\n");
for i = 1:size(X_errors, 1)
	display_data(X_errors(i, :));
	predicted_value = y_errors(i);
	actual_value = y_actual(i);
	% The value for zero is represented by a 10... fix it.
	if (predicted_value == 10)
		predicted_value = 0;
	endif
	if (actual_value == 10)
		actual_value = 0;
	endif
	printf("Error %3d of %3d: ", i, size(X_errors, 1));
	printf("model predicted %d, actual %d, probability=%4.1f%%\n", predicted_value, actual_value, prob_errors(i)*100.0);
	printf("Hit any key to see next error\n");
	pause;
endfor
