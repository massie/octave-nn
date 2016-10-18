## --*- texinfo -*--
##
## @deftypefn {Function File} {@var{theta} =} theta_init (@var{in_size}, @var{out_size}, @var{epsilon})
##
## Computes a random matrix that is (@var{out_size} X @var{in_size} + 1) in size.
## All values in the matrix will be in the range from [-@var{epsilon}, @var{epsilon}].
##
## @end deftypefn

function theta = theta_init(in_size, out_size, epsilon = 0.12)
	theta = rand(out_size, in_size +1) * 2 * epsilon - epsilon;
endfunction
