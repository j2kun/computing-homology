import numpy
import numpy.linalg


def printMatrix(m):
    for row in m:
        print(str(row))


def rowSwap(n, i, j):
    A = numpy.identity(n)
    A[i][i], A[j][j] = 0, 0
    A[i][j], A[j][i] = 1, 1
    return A


def scaleRow(n, i, c):
    A = numpy.identity(n)
    A[i][i] = c
    return A


def linComb(n, addTo, scaleRow, scaleAmt):
    A = numpy.identity(n)
    A[addTo][scaleRow] = scaleAmt
    return A


'''
X = numpy.array([[1,1,1], [2,2,2], [3,3,3]])
X = rowSwap(3, 0, 2).dot(X)
X = linComb(3, 0, 2, 2).dot(X)
print(X)
'''


def rref(matrix):
    if not matrix: return
    numRows = len(matrix)
    numCols = len(matrix[0])

    basisChange = numpy.identity(numRows)

    i, j = 0, 0
    while True:
        if i >= numRows or j >= numCols:
            break

        if matrix[i][j] == 0:
            nonzeroRow = i
            while nonzeroRow < numRows and matrix[nonzeroRow][j] == 0:
                nonzeroRow += 1

            if nonzeroRow == numRows:
                j += 1
                continue

            temp = matrix[i]
            matrix[i] = matrix[nonzeroRow]
            matrix[nonzeroRow] = temp
            basisChange = rowSwap(numRows, i, nonzeroRow).dot(basisChange)
            print(f"row swap {i} <-> {nonzeroRow}")

        pivot = matrix[i][j]
        matrix[i] = [x / pivot for x in matrix[i]]
        basisChange = scaleRow(numRows, i, 1.0 / pivot).dot(basisChange)
        print(f"scale R {i} by {1.0 / pivot}")

        for otherRow in range(0, numRows):
            if otherRow == i:
                continue
            if matrix[otherRow][j] != 0:
                print(f"row lin comb: R{otherRow} = R{otherRow} - matrix[otherRow][j] * R{i}")
                basisChange = linComb(numRows, otherRow, i, -matrix[otherRow][j]).dot(basisChange)

                matrix[otherRow] = [y - matrix[otherRow][j] * x
                                    for (x, y) in zip(matrix[i], matrix[otherRow])]

        i += 1;
        j += 1

    return matrix, basisChange


bd1 = numpy.array([[-1, -1, -1, -1, 0, 0, 0, 0], [1, 0, 0, 0, -1, -1, 0, 0],
                   [0, 1, 0, 0, 1, 0, -1, -1], [0, 0, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0, 1]])
toReduce = bd1.T.tolist()

rrefbd1T, trans = rref(toReduce)
trans = trans.T
print(f"A is {trans}")

colReduced = bd1.dot(trans)
print(f"col reduced matrix is {colReduced}")

bd2 = numpy.array([[1, 1, 0, 0], [-1, 0, 1, 0], [0, -1, -1, 0],
                   [0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, -1],
                   [0, 0, 1, 1], [0, 0, 0, 0]])

transInv = numpy.linalg.inv(trans)
print(f"inv(A) is {transInv}")

print(f"inv(A) * bd2 is {transInv.dot(bd2)}")
print(f"We still get bd1 * bd2 = 0 after reducing: {colReduced.dot(transInv.dot(bd2))}" )
