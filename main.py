import pypim as pim
def myFunc(a: pim.Tensor, b: pim.Tensor):    
    # Parallel multiplication and addition 
    return a * (1 + b)

# Tensor initialization
x, y = pim.zeros(2 ** 20, dtype=pim.float32), pim.zeros(2 ** 20, dtype=pim.float32)
x[4], y[4] = 8.0, 0.5
x[5], y[5] = 20.0, 1.0
x[8], y[8] = 10.0, 1.0

# Custom function call
z = myFunc(x, y)
# Logarithmic-time reduction of even indices
print(z[::2].sum())  # 32.0 = 8 * 1.5 + 10 * 2