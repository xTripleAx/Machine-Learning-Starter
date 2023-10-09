import numpy as np


#creating a row vector
row_vector = np.array([10,15,6])
print(row_vector)


#creating a column vector
column_vector = np.array([[10],[15],[6]])
print(column_vector)


#create a matrix
matrix = np.array([[10,20,25],[10,15,20]])
print(matrix)


print()


#Accessing Elements
row_vector = np.array([10,20,30,40,50,60])
print("row_vector", row_vector)

matrix = np.array([[10,20,30],[40,50,60],[70,80,90]])
print("matrix", matrix)

#select second element of vector
print("row_vector[1]", row_vector[1])

#select 3rd column 3rd row from matrix
print("matrix[2,2] ", matrix[2,2])

#select all elements of vector
print("row_vector[:]:",row_vector[:])

#select last element of vector
print("row_vector[-1]:", row_vector[-1])

#select first two rows of the matrix
print("matrix[:2,:]:" , matrix[:2,:])


print()


#Matrix
matrix = np.array([[10,20,30],[40,50,60],[70,80,90]])

#print number of rows and columns
print("Rows and Columns:", matrix.shape)

#print number of elements (rows*columns)
print("Total Elements:",matrix.size)

#print number of dimensions
print("Dimension", matrix.ndim)

#print max
print("Max:", np.max(matrix))

#print min
print("Min", np.min(matrix))

#print max of each column
print("Max of column:", np.max(matrix,axis=0))

#print max of each row
print("Max of row:", np.max(matrix,axis=1))

#print average(mean)
print("Mean:",np.mean(matrix))
print("Average:",np.average(matrix))

#reshape
print("Reshaped matrix:",matrix.reshape(9,1))

#reshape to one row and as many columns as needed
print("Reshaped matrix:",matrix.reshape(1,-1))

#flatten an array
print("flattened matrix:" , matrix.flatten())

#transpose the matrix
print(matrix.T)

#print diagonal of the matrix
print(matrix.diagonal())

#Dot Product
vector_one = np.array([10,20,30])
vector_two = np.array([40,50,60])
print(np.dot(vector_one,vector_two))

#adding, subtracting and multiplying
matrix_2 = np.array([[15,25,35],[45,55,65],[75,85,95]])
print(np.add(matrix,matrix_2))
print(matrix+matrix_2)
print(np.subtract(matrix,matrix_2))
print(matrix-matrix_2)

#multiplication element wise not dot product
print(np.multiply(matrix,matrix_2))
print(matrix*matrix_2)

#multiplication row column wise
print(np.matmul(matrix,matrix_2))

#zeros and ones
zeros = np.zeros([4,4])
print(zeros)
ones = np.ones([3,3])
print(ones)

#generate random integers between 1 and 10
print(np.random.randint(0,11,4))

#Draw 3 numbers from a normal distribution with mean 1.0 and std 2.0
print (np.random.normal(1.0, 2.0, 6))