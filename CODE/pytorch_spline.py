import torch

class NaturalCubicSpline_Torch:
    """
    A class for constructing and evaluating a natural cubic spline for a given set of data points.
    The spline is built upon initialization using the provided x and fx values, representing the
    x-coordinates and the corresponding function values, respectively.

    Attributes:
        x (torch.Tensor): The x-coordinates of the data points.
        fx (torch.Tensor): The function values at the data points.
        is_spline_built (bool): Flag indicating whether the spline coefficients have been computed.
        init_cond (torch.Tensor): Initial condition for spline construction, often related to boundary conditions.
        end_cond (torch.Tensor): End condition for spline construction, often related to boundary conditions.
        second_der (torch.Tensor): The second derivatives of the spline at the x-coordinates, used for evaluation.
    """

    def __init__(self, x, fx):
        """
        Initializes the NaturalCubicSpline_Torch class by storing x and fx values, setting initial
        and end conditions, and constructing the spline.

        Parameters:
            x (torch.Tensor): The x-coordinates of the data points.
            fx (torch.Tensor): The function values at the data points.
        """
        self.x = x
        self.fx = fx

        self.is_spline_built = False

        # Default initial and end conditions set to zero, but can be modified if needed
        self.init_cond = torch.tensor(0.0)
        self.end_cond = torch.tensor(0.0)

        # Automatically build the spline upon initialization
        self.build_spline()
        

    def build_system(self):
        """
        Constructs the linear system (A*c = b) required to solve for the spline's second derivatives.

        Returns:
            A (torch.Tensor): The matrix A in the system A*c = b, where c are the coefficients.
            c (torch.Tensor): The vector b in the system A*c = b, representing differences in slope.
        """
        N = len(self.x)
        A = torch.zeros((N-2, N-2), dtype = self.x.dtype)
        c = torch.zeros(N - 2, dtype = self.fx.dtype)

        # Calculate distances between consecutive x values
        h_i_1 = self.x[1:N-1]-self.x[0:N-2]
        h_i = self.x[2:N]-self.x[1:N-1]
        
        # Main diagonal and off-diagonal values of A
        m_i = (h_i + h_i_1) / 3.0
        l_i = h_i_1[1:] / 6.0
        u_i = h_i[:-1] / 6.0

        # Populate the tridiagonal matrix A
        A[range(N-2), range(N-2)] = m_i
        A[range(1,N-2), range(0,N-3)] = l_i
        A[range(0,N-3), range(1,N-2)] = u_i

        # Construct the right-hand side vector c
        c = (self.fx[2:N] - self.fx[1:N-1]) / h_i - (self.fx[1:N-1] - self.fx[0:N-2]) / h_i_1

        return A, c

    def build_spline(self):
        """
        Constructs the spline by solving the linear system for the second derivatives.

        Returns:
            second_der (torch.Tensor): The second derivatives of the spline at the x-coordinates.
        """
        self.is_spline_built = True

        A, c = self.build_system()
        # Boundary conditions are currently set to zero, but this can be adjusted
        bounds = torch.zeros(len(c))

        # Solve the linear system to find the second derivatives
        self.second_der = torch.linalg.solve(A, c - bounds)

        # Prepend and append the initial and end conditions to the second derivatives
        self.second_der = torch.cat((torch.tensor([self.init_cond]), self.second_der, torch.tensor([self.end_cond])))

        return self.second_der

    def evaluate_spline(self, x_i):
        """
        Evaluates the spline at given x_i points.

        Parameters:
            x_i (torch.Tensor): Points at which to evaluate the spline.

        Returns:
            result (torch.Tensor): The evaluated spline values at x_i.
        """
        # Find the right interval for each x_i
        i = torch.searchsorted(self.x, x_i, right=True)
      
        # Clamp values to valid range for index i
        i = torch.clamp(i, min=1, max=len(self.x) - 1)

        # Calculate h_i for each segment
        h_i = self.x[i] - self.x[i - 1]

        # Calculate the first term for each x_i based on spline formula
        first_term = (((self.x[i] - x_i) ** 3) / (6.0 * h_i)) * self.second_der[i - 1] \
                    + ((self.x[i] - x_i) * (self.fx[i - 1] / h_i - (h_i / 6.0) * self.second_der[i - 1]))

        # Calculate the second term for each x_i based on spline formula
        second_term = (((x_i - self.x[i - 1]) ** 3) / (6.0 * h_i)) * self.second_der[i] \
                      + ((x_i - self.x[i - 1]) * (self.fx[i] / h_i - (h_i / 6.0) * self.second_der[i]))

        # The final spline value is the sum of the first and second terms
        # This calculation leverages the piecewise definition of cubic splines
        result = first_term + second_term

        return result

