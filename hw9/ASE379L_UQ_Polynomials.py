'''

 Polynomial utilities for the UQ class


 Polynomial utilities for the UQ class

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
def hermitePoly( x, maxDeg ) :
  '''
  :param x: n-array of input random variables
  :param maxDeg: integer for maximum degree
  :return: polys : (maxDeg+1) x n array of polynomails.  First
                    index corresponds to the degree and the second
                    is the random input
  '''

  polys = np.zeros((maxDeg+1,len(x)))

  polys[0,:] = 1.

  if maxDeg >= 1 :
    polys[1,:] = x

  if maxDeg >= 2 :

    #  Evaluate the three-term recurrence
    for ii in range(2,maxDeg+1) :
      polys[ii,:] = x*polys[ii-1,:] - (ii-1.)*polys[ii-2,:]

    #  Now, we normalize the polynomials
    fact = 1.
    for ii in range(2,maxDeg+1) :
      fact *= float(ii)
      polys[ii,:] /= np.sqrt(fact)

  return polys

#-------------------------------------------------------------------------------
def legendrePoly( x, maxDeg ) :
  '''
  :param x: n-array of input random variables
  :param maxDeg: integer for maximum degree
  :return: polys : (maxDeg+1) x n array of polynomails.  First
                    index corresponds to the degree and the second
                    is the random input
  '''

  polys = np.zeros((maxDeg+1,len(x)))

  polys[0,:] = 1.

  if maxDeg == 1 :
    polys[1,:] = x*np.sqrt(3.)

  if maxDeg >= 2 :

    polys[1,:] = x

    #  Evaluate the three-term recurrence
    for ii in range(2,maxDeg+1) :
      polys[ii,:] = ((2.*float(ii)-1.)*x*polys[ii-1,:] - (ii-1.)*polys[ii-2,:])/float(ii)

    #  Now, we normalize the polynomials
    for ii in range(1,maxDeg+1) :
      polys[ii,:] *= np.sqrt(2.*ii + 1.)

  return polys

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

  print(legendrePoly(np.array([.5]),1))
  print(legendrePoly(np.array([.5]),2))

  #  Example for the Hermite polynomial
  inputs = np.random.randn(10,)
  polyEval = hermitePoly(inputs,5)
  #  Print the degree 5 polynomials (for example)
  print( polyEval[5,:])

  #  Example for the Legendre polynomial
  inputs = np.random.rand(10,)*2 - 1. # Shift to inputs on [-1,1]
  polyEval = legendrePoly(inputs,5)
  #  Print the degree 3 polynomials (for example)
  print( polyEval[3,:])

  return

if __name__ == "__main__":
  main()

