# Brownian motion

The other day we kind of built Brownian motion from a discrete process. For the process to be continuous, we made the variance to depend on $\Delta t^{\gamma}$ and saw that unless $\gamma = 1/2$ the process variance either expoded or vanished as $\Delta t \to 0$. Why is this the case? 

# Itô's Lemma

With the previous result, and with the help of the central limit theorem, we concluded that

$$W_T-W_t \sim N(0,\sqrt{T-t})$$

After that, we approximated the following integral with a discrete sum

$$\int_{t}^T\left(dW_s\right)^2\approx \sum_{j=1}^n\Delta W_j^2$$

and saw that as $n\to\infty$

$$\sum_{j=1}^n\Delta W_j^2 \to \left(T-t\right)$$

That is, the integral, although comprised of stochastic terms, shos no variance.

Why is this the case?



