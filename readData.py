import scipy.sparse as sp

n_users = 943
n_items = 1682
def readData():
	matrix = sp.lil_matrix((n_users,n_items))
	f = open ('u.data', 'r') # user id | item id | rating | timestamp.
	for line in f:
		lst = line.split("\t", 3)

		usr = int(lst[0])-1
		itm = int(lst[1])-1
		matrix[usr, itm] = int(lst[2])
	matrix = matrix.todense()
	return matrix

def main():
	readData()

if __name__ == "__main__":
	main()