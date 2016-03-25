def calcE(data, centers):
	diffsq = (centers[:,np.newaxis,:] - data) ** 2
	return np.sum(np.min(np.sum(diffsq, axis = 2), axis = 0))

def kmeans(data, k = 2, n = 5):

	# initialize centers and list eigenvalues to track performance metric
	centers = data[np.random.choice(range(data.shape[0]), k, replace = False), :]
	E = []

	# repeat n times
	for iteration in range(n):

		# which center is each sample closest to?
		sqdistances = np.sum((centers[:, np.newaxis,:] -  data) ** 2, axis = 2)
		closest = np.argmin(sqdistances, axis = 0)

		# calculate E and append to list E
		E.append(calcE(data, centers))

		# update cluster centers
		for i in range(k):
			centers[i,:] = data[closest==i,:].mean(axis=0)

	# calculate E one final time and return results
	E.append(calcE(data, centers))
	return centers, E, closest