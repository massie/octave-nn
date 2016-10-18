## --*- texinfo -*--
##
## @deftypefn {Function File} {@var{[theta1, theta2]} =} nn_train (@var{X}, @var{y}, @var{desired_error}, @var{max_iterations}, @var{epsilon}, @var{hidden_nodes})
##
## Trains an artificial neural network with a single hidden layer. Each column in @var{X} is a feature and each
## row in @var{X} is an observation. The @var{y} value contains the classifications for each observation. For 
## multiclassification problems, @var{y} will have more than one column. After training, this function returns
## the calculated @var{[theta1, theta2]} that can be used for predictions.
##
## The number of input nodes in the neural network is set to the number of columns (features) in @var{X}. The number
## of output nodes is set to the number of columns (classifcations) in @var{y}. The number of hidden nodes is set
## to @code{floor(input_nodes * 2/ 3 + output_nodes)}, unless explicitly provided.
##
## The training will end when the @var{desired_error} or @var{max_iterations} is reached whichever comes first.
## The default value for @var{max_iterations} is 100000. The default value for @var{epsilon} is 0.12. 
##
## The neural network uses a sigmoid activation function.
##
## @seealso{nn_predict}
## @end deftypefn
##
function [theta1, theta2] = nn_train(X, y, desired_error, max_iterations = 100000, epsilon = 0.12, hidden_nodes = 0)

	m = size(X, 1);
	input_nodes = size(X, 2);
	output_nodes = size(y, 2);
	if (hidden_nodes <= 0)
		hidden_nodes = floor(input_nodes * 2 / 3 + output_nodes);
	endif
	theta1 = theta_init(input_nodes, hidden_nodes, epsilon)';
	theta2 = theta_init(hidden_nodes, output_nodes, epsilon)';

	% Move constants outside of the loop
	% The first activation layer is constant
	a1 = [ones(size(X, 1), 1) X];
	% The bias unit ones are constant too
	a2_ones = ones(size(a1, 1), 1);

	printf("Training the neural network (%d input, %d hidden, %d output nodes) with %d observations\n", ...
			input_nodes, hidden_nodes, output_nodes, m);

	% String used in writing updates to the terminal
	s=""; 

	% Set up internal timer before going into the loop
	tic_id = tic();

	for k = 1:max_iterations
		% Feed forward
		a2 = [a2_ones sigmoid( a1 * theta1 )];
		a3 = sigmoid( a2 * theta2 );

		a3_delta = y - a3;

		% Each second report the current state to the user
		if (toc(tic_id) > 1)
			meansq_error = mean(meansq(a3_delta));
			printf(repmat("\b", [1,length(s)]))
			printf(s = sprintf("Iteration: %9d (max:%d), mse: %9f (target:%f)", ...
			                   k, max_iterations, meansq_error, desired_error));
			tic_id = tic();
			if (meansq_error < desired_error)
				break
			endif
		endif
	
		% Backpropagation
		a2_error = a3_delta * theta2';
		a2_delta = a2_error .* sigmoid(a2, true);

		theta2 += ((a2' * a3_delta) ./ m);
		theta1 += ((a1' * a2_delta) ./ m)(:, 2:end);
	endfor

endfunction
