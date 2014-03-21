import numpy as np
import theano
import theano.tensor as T 

theano.config.warn.subtensor_merge_bug = False 

coef = T.vector('coef')
x = T.scalar('x')
max_coef_supported = 10000

# generate the components of the polynomial
full_range = T.arange(max_coef_supported)
components, updates = theano.scan(fn=lambda coeff, power, free_var: coeff * (free_var ** power), 
 								outputs_info = None, 
								sequences = [coef, full_range],
								non_sequences=x)

polynomial = components.sum()
calculate_polynomial = theano.function(inputs = [coef, x], outputs=polynomial)

test_data = np.asarray([1, 0, 2], dtype=np.float32)
print calculate_polynomial(test_data, 3)