'''

 Compute PCE multi-indices


 Code to compute multi-indices for multi-variate PCEs

'''

__author__ = 'Brandon A. Jones'

################################################################################
#                     I M P O R T     L I B R A R I E S
################################################################################

import numpy as np

################################################################################
#                    E X P O R T E D     C L A S S E S:
################################################################################

#-------------------------------------------------------------------------------


################################################################################
#                  E X P O R T E D     F U N C T I O N S:
################################################################################

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def getMultiIndex( deg, dim ) :
  '''
  Get all possible combinations of polynomial sums that yield a maximum
  degree of deg and use a multivariate basis of dimension dim

  Inputs
  ---------
  deg - (int) maximum degree of each element of the multi-index
  dim - (int) dimension of each element of the multi-index

  Outputs
  ---------
  scheme - (tuple) tuple of tuples defining the set of multi-indices

  '''

  #  Define the sort function used to decide the order of the multiindices
  def sortFcn( a, b ) :
    asum = 0
    bsum = 0
    for i in range( 0, len(a) ) :
      asum += a[i]
      bsum += b[i]
    if asum > bsum : return 1
    elif asum < bsum : return -1
    else :
      if max(a) > max(b) : return 1
      elif max(a) < max(b) : return -1
      for i in range( 0, len(a) ) :
        if a[i] > b[i] : return -1
        elif a[i] < b[i] : return 1
        else : continue

  #
  #  The scheme has not been generated yet.  Here, we generate the set of
  #  multiindices.  For this, we use the algorithm presented in "Spectral
  #  Methods for Uncertainty Quantification with Applications to Computational
  #  Fluid Dynamics" by Olivier P. LeMaitre and Omar M. Knio, pp. 516-517.
  #

  #  Set the element for the 0-degree polynomials
  multiIndices = [[0]*dim]

  #  Set the elements for the polynomials of degree 1
  if deg >= 1 :
    for i in range( dim ) :
      newElem = [0]*dim
      newElem[i] = 1
      multiIndices.append( newElem )

  #  Now, we generate the elements for higher degree polynomials.  This
  #  algorithm is based on the pseudocode in the above reference.
  P = dim
  Pi_p = [1]*dim
  Pi = [0]*dim
  for k in range( 2, deg+1 ) :
    L = P
    Pi = [0]*dim
    for i in range( 0, dim ) :
      for m in range( i, dim ) :
        Pi[i] += Pi_p[m]
    for j in range( 0, dim ) :
      for m in range( L-Pi[j]+1, L+1 ) :
        multiIndices.append([0]*dim)
        P += 1
        for i in range( 0, dim ) :
          multiIndices[P][i] = multiIndices[m][i]
        multiIndices[-1][j] += 1
    Pi_p = list( Pi )

  #  Now that we have the sorted set, we now convert the elements from list
  #  to tuples.  This ensures that it cannot be altered by any other means.
  scheme = []
  for set in multiIndices :
    scheme.append( tuple(set) )

  #  Done!
  return scheme


################################################################################
#             U N I T     T E S T     C A S E     F U N C T I O N:
################################################################################


################################################################################
#                       M A I N     F U N C T I O N:
################################################################################

def main():
  '''
  <Description>

  Parameters
  ----------
  inputs : description

  Returns
  -------
  outputs : description

  Examples
  --------
  Examples, if needed
  '''

  return

if __name__ == "__main__":
  main()

