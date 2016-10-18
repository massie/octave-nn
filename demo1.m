clear; close all; clc;

X = [0 0; 0 1; 1 0; 1 1];
y = [0; 1; 1; 0];

printf("*** XOR Demo ***\n");

[theta1, theta2] = nn_train(X, y, 0.0001);

pred_values = nn_predict(X, theta1, theta2);
printf("\n\n      Input Values   Predicted   Actual\n");
disp([X pred_values y])
printf("\nMean square error of trained model predictions: %f\n", mean(meansq(y - pred_values)))
