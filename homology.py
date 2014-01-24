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

   numRows, numCols = A.shape

   i,j = 0,0
   while True:
      if i >= numRows or j >= numCols:
         break

      if A[i,j] == 0:
         nonzeroCol = j
         while nonzeroCol < numCols and A[i,nonzeroCol] == 0:
            nonzeroCol += 1

         if nonzeroCol == numCols:
            i += 1
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


def finishRowReducing(B):
   numRows, numCols = B.shape

   i,j = 0,0
   while True:
      if i >= numRows or j >= numCols:
         break

      if B[i, j] == 0:
         nonzeroRow = i
         while nonzeroRow < numRows and B[nonzeroRow, j] == 0:
            nonzeroRow += 1

         if nonzeroRow == numRows:
            j += 1
            continue

         rowSwap(B, i, nonzeroRow)

      pivot = B[i, j]
      scaleRow(B, i, 1.0 / pivot)

      for otherRow in range(0, numRows):
         if otherRow == i:
            continue
         if B[otherRow, j] != 0:
            scaleAmt = -B[otherRow, j]
            rowCombine(B, otherRow, j, scaleAmt)

      i += 1; j+= 1

   return B


def numPivotCols(A):
   z = numpy.zeros(A.shape[0])
   return [numpy.all(A[:, j] == z) for j in range(A.shape[1])].count(False)


def numPivotRows(A):
   z = numpy.zeros(A.shape[1])
   return [numpy.all(A[i, :] == z) for i in range(A.shape[0])].count(False)


def bettiNumber(d_k, d_kplus1):
   A, B = numpy.copy(d_k), numpy.copy(d_kplus1)
   simultaneousReduce(A, B)
   finishRowReducing(B)

   dimKChains = A.shape[1]
   print(dimKChains)
   kernelDim = dimKChains - numPivotCols(A)
   print(kernelDim)
   imageDim = numPivotRows(B)
   print(imageDim)

   return kernelDim - imageDim


bd0 = numpy.array([[0,0,0,0,0]])
bd1 = numpy.array([[-1,-1,-1,-1,0,0,0,0], [1,0,0,0,-1,-1,0,0],
     [0,1,0,0,1,0,-1,-1], [0,0,1,0,0,1,1,0], [0,0,0,1,0,0,0,1]])
bd2 = numpy.array([[1,1,0,0],[-1,0,1,0],[0,-1,-1,0],
     [0,0,0,0],[1,0,0,1],[0,1,0,-1],
     [0,0,1,1],[0,0,0,0]])
bd3 = numpy.array([[-1],[1],[-1],[1]])


print("Example complex from post")
print("0th homology: %d" % bettiNumber(bd0,bd1))
print("1st homology: %d" % bettiNumber(bd1,bd2))
print("2nd homology: %d" % bettiNumber(bd2,bd3))


mobiusD1 = numpy.array([
   [-1,-1,-1,-1, 0, 0, 0, 0, 0, 0],
   [ 1, 0, 0, 0,-1,-1,-1, 0, 0, 0],
   [ 0, 1, 0, 0, 1, 0, 0,-1,-1, 0],
   [ 0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
])

mobiusD2 = numpy.array([
   [ 1, 0, 0, 0, 1],
   [ 0, 0, 0, 1, 0],
   [-1, 0, 0, 0, 0],
   [ 0, 0, 0,-1,-1],
   [ 0, 1, 0, 0, 0],
   [ 1,-1, 0, 0, 0],
   [ 0, 0, 0, 0, 1],
   [ 0, 1, 1, 0, 0],
   [ 0, 0,-1, 1, 0],
   [ 0, 0, 1, 0, 0],
])

print("Mobius Band")
print("1st homology: %d" % bettiNumber(mobiusD1, mobiusD2))
