import numpy
import numpy.linalg

def rowSwap(A, i, j):
   temp = numpy.copy(A[i, :])
   A[i, :] = A[j, :]
   A[j, :] = temp

def colSwap(A, i, j):
   temp = numpy.copy(A[:, i])
   A[:, i] = A[:, j]
   A[:, j] = temp

def scaleCol(A, i, c):
   A[:, i] *= c*numpy.ones(A.shape[0])

def scaleRow(A, i, c):
   A[i, :] *= c*numpy.ones(A.shape[1])

def colCombine(A, addTo, scaleCol, scaleAmt):
   A[:, addTo] += scaleAmt * A[:, scaleCol]

def rowCombine(A, addTo, scaleRow, scaleAmt):
   A[addTo, :] += scaleAmt * A[scaleRow, :]

def simultaneousReduce(A, B):
   if A.shape[1] != B.shape[0]:
      raise Exception("Matrices have the wrong shape.")

   numRows, numCols = A.shape # col reduce A

   i,j = 0,0
   while True:
      if i >= numRows or j >= numCols:
         break

      if A[i][j] == 0:
         nonzeroCol = j
         while nonzeroCol < numCols and A[i,nonzeroCol] == 0:
            nonzeroCol += 1

         if nonzeroCol == numCols:
            j += 1
            continue

         colSwap(A, j, nonzeroCol)
         rowSwap(B, j, nonzeroCol)

      pivot = A[i,j]
      scaleCol(A, j, 1.0 / pivot)
      scaleRow(B, j, 1.0 / pivot)

      for otherCol in range(0, numCols):
         if otherCol == j:
            continue
         if A[i, otherCol] != 0:
            scaleAmt = -A[i, otherCol]
            colCombine(A, otherCol, j, scaleAmt)
            rowCombine(B, j, otherCol, -scaleAmt)

      i += 1; j+= 1

   return A,B


def numPivotCols(A):
   z = numpy.zeros(A.shape[0])
   return [numpy.all(A[:, j] == z) for j in range(A.shape[1])].count(False)


def numPivotRows(A):
   z = numpy.zeros(A.shape[1])
   return [numpy.all(A[i, :] == z) for i in range(A.shape[0])].count(False)


def bettiNumber(d_k, d_kplus1):
   A, B = numpy.copy(d_k), numpy.copy(d_kplus1)
   simultaneousReduce(A, B)

   dimKChains = A.shape[1]
   kernelDim = dimKChains - numPivotCols(A)
   imageDim = numPivotRows(B)

   return kernelDim - imageDim


bd1 = numpy.array([[-1,-1,-1,-1,0,0,0,0], [1,0,0,0,-1,-1,0,0],
     [0,1,0,0,1,0,-1,-1], [0,0,1,0,0,1,1,0], [0,0,0,1,0,0,0,1]])

bd2 = numpy.array([[1,1,0,0],[-1,0,1,0],[0,-1,-1,0],
     [0,0,0,0],[1,0,0,1],[0,1,0,-1],
     [0,0,1,1],[0,0,0,0]])

#simultaneousReduce(bd1, bd2)
print bd1
print bd2
print bettiNumber(bd1, bd2)
