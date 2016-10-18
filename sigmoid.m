## -*- texinfo -*-
##
## @deftypefn {Function File} {@var{y} =} sigmoid (@var{x}, @var{derivative})
##
## Computes the sigmoid of @var{x} or the sigmoid derivative of @var{x} if
## @var{derivative} is set to true
##
## @end deftypefn
function y = sigmoid(x, derivative=false)
	if (derivative) 
		y = x.*(1-x);
	else
		y = 1.0 ./ (1.0 + exp(-x));
	endif
endfunction
%!assert(sigmoid(0), 0.5)
%!assert(sigmoid([0]), [0.5])
%!assert(sigmoid(1), 0.73105, 0.00001)
%!assert(sigmoid([0, 1]), [0.5, 0.73105], 0.00001)
%!assert(sigmoid([0 1; 0 1]), [0.5, 0.73105; 0.5, 0.73105], 0.00001)
%!assert(sigmoid(0.2, true), 0.16, 0.00001)
