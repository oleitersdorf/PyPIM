import math

def degrees_to_radians(x):
    """
    Converts the given Tensor from degrees to radians.
    """
    return x * math.pi / 180

def radians_to_degrees(x):
    """
    Conversts the given Tensor from radians to degrees.
    """

    return x * 180 / math.pi

def cos_sin(alpha, num_iters):
    """
    Computes the cosine and sine of the given vector of angles (radians)
    """

    theta, x, y = alpha.zeros_like(), alpha.ones_like(), alpha.zeros_like()

    # Compute constants
    K_n = math.prod([1 / math.sqrt(1 + 2 ** (-2 * i)) for i in range(num_iters)])

    # CORDIC iterations
    for i in range(num_iters):

        sigma = theta < alpha

        arc_tangent = math.atan2(1, 2**i)  # constant for entire Tensor
        theta = sigma.mux(theta + arc_tangent, theta - arc_tangent)
        x, y = x - sigma.mux(y, -y) * (2 ** -i), y + sigma.mux(x, -x) * (2 ** -i)
    
    return x * K_n, y * K_n

def cos(alpha, num_iters=32):
    """
    Computes the cosine of the given vector of angles (radians in range [-pi, pi])
    """
    return cos_sin(alpha, num_iters)[0]

def sin(alpha, num_iters=32):
    """
    Computes the sine of the given vector of angles (radians in range [-pi, pi])
    """
    return cos_sin(alpha, num_iters)[1]

def tan(alpha, num_iters=32):
    """
    Computes the sine of the given vector of angles (radians in range [-pi, pi])
    """
    cs = cos_sin(alpha, num_iters)
    return cs[1] / cs[0]