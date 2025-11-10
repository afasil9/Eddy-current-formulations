from ufl import cos, pi, as_vector, sin


def sinusoidal(x):
    return as_vector(
        (
            cos(pi * x[1]),
            cos(pi * x[2]),
            cos(pi * x[0]),
        )
    )

def sinusoidal_time(x, t):
    return as_vector(
        (
            cos(pi * x[1]) * sin(pi * t),
            cos(pi * x[2]) * sin(pi * t),
            cos(pi * x[0]) * sin(pi * t),
        )
    )

def quadratic(x):
    return as_vector(
        (
            x[1]**2,
            x[2]**2,
            x[0]**2,
        )
    )

def quadratic_time(x, t):
    return as_vector((
        x[1]**2 + x[0] * t, 
        x[2]**2 + x[1] * t, 
        x[0]**2 + x[2] * t))