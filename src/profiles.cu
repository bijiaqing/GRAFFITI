#include "cudust.cuh"

// =========================================================================================================================

__host__
void rand_uniform (real *profile, int number, real p_min, real p_max)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    for (int i = 0; i < number; i++)
    {
        profile[i] = p_min + (p_max - p_min)*random(rand_generator);
    }
}

// =========================================================================================================================

__host__
void rand_gaussian (real *profile, int number, real p_min, real p_max, real p_0, real sigma)
{
    std::normal_distribution <real> random(p_0, sigma);

    for (int i = 0; i < number; i++)
    {
        real value;
        
        do
        {
            value = random(rand_generator);
        } 
        while (value < p_min || value > p_max);
        
        profile[i] = value;
    }
}

// =========================================================================================================================

__host__
void rand_pow_law (real *profile, int number, real p_min, real p_max, real idx_pow)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    real tmp_min = std::pow(p_min, idx_pow + 1.0);
    real tmp_max = std::pow(p_max, idx_pow + 1.0);

    // check https://mathworld.wolfram.com/RandomNumber.html for derivations
    // NOTE: this is the probability distribution function dN(x) ~ x^n*dx
    for (int i = 0; i < number; i++)
    {
        profile[i] = std::pow((tmp_max - tmp_min)*random(rand_generator) + tmp_min, 1.0/(idx_pow + 1.0));
    }
}

// =========================================================================================================================

__host__
real gaussian (real x, real x_0, real sigma)
{
    return std::exp(-(x - x_0)*(x - x_0)/(2.0*sigma*sigma));
}

__host__
real tapered_pow (real x, real x_min, real x_max, real idx_pow)
{
    real output;

    if (x >= x_min && x <= x_max)
    {
        output = std::pow(x / x_min, idx_pow);
    }
    else
    {
        output = 0.0;
    }

    return output;
}

__host__
void rand_convpow (real *profile, int number, real x_min, real x_max, real idx_pow, real smooth, int bins)
{
    // a convolved (i.e., smoothed) power-law profile
    
    // x_min and x_max define the hard boundary of the smoothed profile
    // p_min and p_max define the domain that is not too much smoothed
    real p_min = x_min + 2.0*smooth;
    real p_max = x_max - 2.0*smooth;
    
    // Build x-axis and compute convolved profile
    std::vector<real> x_axis(bins + 1);
    std::vector<real> y_axis(bins + 1, 0.0);
    
    real dx = (x_max - x_min) / static_cast<real>(bins);
    
    for (int i = 0; i < bins + 1; i++)
    {
        x_axis[i] = x_min + i*dx;
    }

    // Convolve tapered power law with Gaussian kernel
    for (int j = 0; j < bins + 1; j++)
    {
        for (int k = 0; k < bins + 1; k++)
        {
            y_axis[k] += tapered_pow(x_axis[j], p_min, p_max, idx_pow)*gaussian(x_axis[k], x_axis[j], 0.5*smooth);
        }
    }

    // Find peak for rejection sampling
    real y_max = *std::max_element(y_axis.begin(), y_axis.end());

    // Generate random numbers via rejection sampling
    std::uniform_real_distribution<real> random(0.0, 1.0);
    
    real rand_x, rand_y, frac_x, real_y;
    
    int idx;
    
    for (int n = 0; n < number; n++)
    {
        do
        {
            rand_x = x_min + (x_max - x_min)*random(rand_generator);
            rand_y = y_max                  *random(rand_generator);
            
            // Linear interpolation of y_axis at rand_x
            idx = static_cast<int>((rand_x - x_min) / dx);
            frac_x = (rand_x - x_min) / dx - idx;
            real_y = (1.0 - frac_x)*y_axis[idx] + frac_x*y_axis[idx + 1];
        }
        while (rand_y > real_y); // only accept if under the curve
        
        profile[n] = rand_x;
    }
}