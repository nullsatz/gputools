#define NTHREADS 16

__global__ void scale(real knot_max, int nx, int nsamples,
                      real * x, int pitch_x)
{
  int
    col_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(col_idx >= nx) return;

  real
    min, max,
    * col = x + col_idx * pitch_x;

  // find the min and the max
  min = max = col[0];
  for(int i = 1; i < nsamples; i++) {
    if(col[i] < min) min = col[i];
    if(col[i] > max) max = col[i];
  }

  real delta = max - min;
  for(int i = 0; i < nsamples; i++)
    col[i] = (knot_max * (col[i] - min)) / delta;
}

__device__ real do_fraction(real numer, real denom) {
  real result = real(0); 

  if((numer == denom) && (numer != real(0)))
    result = real(1);
  else if(denom != real(0))
    result = numer / denom;

  return result;
}

// bins must be initialized to zero before calling get_bin_scores
__global__ void get_bin_scores(int nbins, int order,
                               int nknots, real * knots, int nsamples,
                               int nx, real * x, int pitch_x,
                               real * bins, int pitch_bins)
{
  int
    col_x = blockDim.x * blockIdx.x + threadIdx.x;

  if(col_x >= nx)
    return;

  real
    ld, rd, z,
    term1, term2,
    * in_col = x + col_x * pitch_x,
    * bin_col = bins + col_x * pitch_bins;
  int i0;

  for(int k = 0; k < nsamples; k++, bin_col += nbins) {
    z = in_col[k];
    i0 = (int)floor(z) + order - 1;
    if(i0 >= nbins)
      i0 = nbins - 1;

    bin_col[i0] = real(1);
    for(int i = 2; i <= order; i++) {
      for(int j = i0 - i + 1; j <= i0; j++) {
        rd = do_fraction(knots[j + i] - z, knots[j + i] - knots[j + 1]);

        if((j < 0) || (j >= nbins) || (j >= nknots) || (j + i - 1 < 0) || (j > nknots))
          term1 = real(0);
        else {
          ld = do_fraction(z - knots[j],
                           knots[j + i - 1] - knots[j]);
          term1 = ld * bin_col[j];
        }

        if((j + 1 < 0) || (j + 1 >= nbins) || (j + 1 >= nknots) || (j + i < 0) || (j + i >= nknots))
          term2 = real(0);
        else {
          rd = do_fraction(knots[j + i] - z,
                           knots[j + i] - knots[j + 1]);
          term2 = rd * bin_col[j + 1];
        }
        bin_col[j] = term1 + term2;
      }
    }
  }
}

__global__ void get_entropy(int nbins, int nsamples, int nx,
                            real * bin_scores, int pitch_bin_scores, real * entropies)
{
  int
    col_x = blockDim.x * blockIdx.x + threadIdx.x;

  if(col_x >= nx)
    return;

  real
    * in_col = bin_scores + col_x * pitch_bin_scores,
    entropy = real(0), prob, logp;

  for(int i = 0; i < nbins; i++) {
    prob = real(0);
    for(int j = 0; j < nsamples; j++)
      prob += in_col[j * nbins + i];
    prob /= (double) nsamples;

    if(prob <= real(0))
      logp = real(0);
    else
      logp = log2(prob);

    entropy += prob * logp;
  }
  entropies[col_x] = -entropy;
}

__global__ void get_mi(int nbins, int nsamples,
                       int nx, real * x_bin_scores, int pitch_x_bin_scores,
                       real * entropies_x,
                       int ny, real * y_bin_scores, int pitch_y_bin_scores,
                       real * entropies_y,
                       real * mis, int pitch_mis)
{
  int
    col_x = blockDim.x * blockIdx.x + threadIdx.x,
    col_y = blockDim.y * blockIdx.y + threadIdx.y;

  if((col_x >= nx) || (col_y >= ny))
    return;

  real
    prob, logp, mi = real(0),
    * x_bins = x_bin_scores + col_x * pitch_x_bin_scores,
    * y_bins = y_bin_scores + col_y * pitch_y_bin_scores;
                
  // calculate joint entropy
  for(int i = 0; i < nbins; i++) {
    for(int j = 0; j < nbins; j++) {
      prob = real(0);
      for(int k = 0; k < nsamples; k++)
        prob += x_bins[k * nbins + i] * y_bins[k * nbins + j];
      prob /= (real)nsamples;

      if(prob <= real(0))
        logp = real(0);
      else
        logp = log2(prob);

      mi += prob * logp;
    }
  }

  // calculate mi from entropies
  mi += entropies_x[col_x] + entropies_y[col_y];
  (mis + col_y * pitch_mis)[col_x] = mi;
}
