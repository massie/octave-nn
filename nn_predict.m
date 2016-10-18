## --*- texinfo -*--
##
## @deftypefn {Function File} {@var{a3} =} nn_predict (@var{X}, @var{theta1}, @var{theta2})
##
## Uses @var{theta1} and @var{theta2} to calculate the predicted outputs for the input @var{X}.
##
## @seealso nn_train
##
## @end deftypefn
##
function a3 = nn_predict(X, theta1, theta2)
	a2 = sigmoid([ones(size(X, 1), 1) X] * theta1);
	a3 = sigmoid([ones(size(X, 1), 1) a2] * theta2);
endfunction
