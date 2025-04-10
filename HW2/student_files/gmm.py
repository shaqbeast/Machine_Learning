import numpy as np
from kmeans import KMeans
from numpy.linalg import LinAlgError
from tqdm import tqdm
SIGMA_CONST = 1e-06
LOG_CONST = 1e-32
FULL_MATRIX = True


class GMM(object):

	def __init__(self, X, K, max_iters=100):
		"""
		Args:
			X: the observations/datapoints, N x D numpy array
			K: number of clusters/components
			max_iters: maximum number of iterations (used in EM implementation)
		"""
		self.points = X
		self.max_iters = max_iters
		self.N = self.points.shape[0]
		self.D = self.points.shape[1]
		self.K = K
		self.num_iters = 1

	def softmax(self, logit):
		"""		
		Args:
			logit: N x D numpy array
		Return:
			prob: N x D numpy array. See the above function.
		Hint:
			Add keepdims=True in your np.sum() function to avoid broadcast error.
		"""
		max_value_row = np.max(logit, axis=1, keepdims=True) # an N x 1 numpy array that has the max values in each row
		subtracted_logit = logit - max_value_row # use broadcasting to subtract each value in the rows from max 

		prob = np.exp(subtracted_logit) / np.sum(np.exp(subtracted_logit), axis=1, keepdims=True)

		return prob

	def logsumexp(self, logit):
		"""		
		Args:
			logit: N x D numpy array
		Return:
			s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
		Hint:
			The keepdims parameter could be handy
		"""
		max_value_row = np.max(logit, axis=1, keepdims=True)
		subtracted_logit = logit - max_value_row
		exp_logit = np.exp(subtracted_logit)
  
		s = np.log(np.sum(exp_logit, axis=1, keepdims=True) + LOG_CONST)
		s += max_value_row

		return s 

	def normalPDF(self, points, mu_i, sigma_i):
		"""		
		Args:
			points: N x D numpy array
			mu_i: (D,) numpy array, the center for the ith gaussian.
			sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
		Return:
			pdf: (N,) numpy array, the probability density value of N data for the ith gaussian
		
		Hint:
			np.diagonal() should be handy.
		"""
		raise NotImplementedError

	def multinormalPDF(self, points, mu_i, sigma_i):
		"""		
		Args:
			points: N x D numpy array
			mu_i: (D,) numpy array, the center for the ith gaussian.
			sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
		Return:
			normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian
		
		Hint:
			1. np.linalg.det() and np.linalg.inv() should be handy.
			2. Note the value in self.D may be outdated and not correspond to the current dataset.
			3. You may wanna check if the matrix is singular before implementing calculation process.
		"""
		  # Calculate Normalization Constant
		_, D = points.shape
		det_sigma_i = np.linalg.det(sigma_i) 
		normalization_constant = 1 / (np.sqrt((2 * np.pi) ** D * det_sigma_i))
  
		# Calculate Mahalanobis Term
		x_minus_mu = points - mu_i # N x D 
		try:
			inverted_sigma_i = np.linalg.inv(sigma_i)
		except LinAlgError:
			sigma_i += sigma_i + SIGMA_CONST
			inverted_sigma_i = np.linalg.inv(sigma_i)
			det_sigma_i = np.linalg.det(sigma_i) 
		mahalanobis_term = -0.5 * (np.sum(((x_minus_mu @ inverted_sigma_i) * x_minus_mu), axis=1)) # not sure how this computation is working

		# Calculate Final Value
		normal_pdf = normalization_constant * np.exp(mahalanobis_term)
		return normal_pdf

	def create_pi(self):
		"""		
		Initialize the prior probabilities
		Args:
		Return:
		pi: numpy array of length K, prior
		"""
		pi = np.zeros(self.K)
		array_indices = np.arange(self.K)
		pi[array_indices] = 1 / self.K

		return pi

	def create_mu(self):
		"""		
		Intialize random centers for each gaussian
		Args:
		Return:
		mu: KxD numpy array, the center for each gaussian.
		"""
		initial_means_indices = np.random.choice(self.N, self.K, replace=True) # out of N points, select K random indices
		mu = self.points[initial_means_indices]
  
		return mu 

	def create_mu_kmeans(self, kmeans_max_iters=1000, kmeans_rel_tol=1e-05):
		"""
		Intialize centers for each gaussian using your KMeans implementation from Q1
		Args:
		Return:
		mu: KxD numpy array, the center for each gaussian.
		"""
		kmeans = KMeans(self.points, self.K, max_iters=kmeans_max_iters, rel_tol=kmeans_rel_tol)
		mu = kmeans.init_centers()
	
		return mu

	def create_sigma(self):
		"""		
		Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the
		by K diagonal matrices.
		Args:
		Return:
		sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
			You will have KxDxD numpy array for full covariance matrix case
		"""
		sigma = np.zeros((self.K, self.D, self.D))

		for k in range(self.K):
			sigma[k] = np.eye(self.D)
		
		return sigma 

	def _init_components(self, kmeans_init=False, **kwargs):
		"""		
		Args:
			kwargs: any other arguments you want
		Return:
			pi: numpy array of length K, prior
			mu: KxD numpy array, the center for each gaussian.
			sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
				You will have KxDxD numpy array for full covariance matrix case
		
			Hint: np.random.seed(5) must be used at the start of this function to ensure consistent outputs.
			Note: If you have used random initialization for centers, invoke create_mu(), else if you have used
			your KMeans implementation from Q1 for center initialization, invoke create_mu_kmeans().
		"""
		np.random.seed(5)
		pi = self.create_pi()
		if kmeans_init:
			mu = self.create_mu_kmeans()
		else:
			mu = self.create_mu()
		sigma = self.create_sigma()
		
		return (pi, mu, sigma)

	def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
		"""		
		Args:
			pi: np array of length K, the prior of each component
			mu: KxD numpy array, the center for each gaussian.
			sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
			array for full covariance matrix case
			full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
		
		Return:
			ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
		"""
		ll = np.zeros((self.N, self.K))
		for k in range(self.K):
			log_pi_k = np.log(pi[k] + LOG_CONST)
			log_pdf_k = np.log(self.multinormalPDF(self.points, mu[k], sigma[k]) + LOG_CONST)
			log_likelihood_k = log_pi_k + log_pdf_k
			ll[:, k] = log_likelihood_k


		return ll 

	def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
		"""		
		Args:
			pi: np array of length K, the prior of each component
			mu: KxD numpy array, the center for each gaussian.
			sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
			array for full covariance matrix case
			full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
		Return:
			tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
		
		Hint:
			You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
		"""
		logit = self._ll_joint(pi=pi, mu=mu, sigma=sigma)
		tau = self.softmax(logit)

		return tau 

	def _M_step(self, tau, full_matrix=FULL_MATRIX, **kwargs):
		"""		
		Args:
			tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
			full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
		Return:
			pi: np array of length K, the prior of each component
			mu: KxD numpy array, the center for each gaussian.
			sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
			array for full covariance matrix case
		
		Hint:
			There are formulas in the slides and in the Jupyter Notebook.
			Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
		"""
		# pi 
		pi = np.sum(tau, axis=0) / self.N # column-wise sum of the soft cluster assignments of tau and then normalizing by dividing by number of points

		# mu
		mu = np.zeros((self.K, self.D)) # create array 
		for k in range(self.K):
			numerator = tau[:, k][:, None] * self.points # self.points is N x D
			mean = np.sum(numerator, axis=0) / np.sum(tau[:, k])
			mu[k] = mean
   
		# sigma
		sigma = np.zeros((self.K, self.D, self.D))
		for k in range(self.K):
			centered_points = self.points - mu[k] # deviation of each point from mu[k]
			cov = (tau[:, k] * (centered_points).T @ centered_points) / np.sum(tau[:, k])
			sigma[k] = cov

		return (pi, mu, sigma)

	def __call__(self, full_matrix=FULL_MATRIX, kmeans_init=False, rel_tol=
		1e-16, **kwargs):
		"""		
		Args:
			rel_tol: convergence criteria w.r.t relative change of loss
			kwargs: any additional arguments you want
		
		Return:
			tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
			(pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)
		
		Hint:
			You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
		"""
		pi, mu, sigma = self._init_components(kmeans_init, **kwargs)
		pbar = tqdm(range(self.max_iters))

		prev_loss = None
		for it in pbar:
			# E-step
			tau = self._E_step(pi, mu, sigma, full_matrix)

			# M-step
			pi, mu, sigma = self._M_step(tau, full_matrix)

			# calculate the negative log-likelihood of observation
			joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
			loss = -np.sum(self.logsumexp(joint_ll))
			if it:
				diff = np.abs(prev_loss - loss)
				if diff / prev_loss < rel_tol:
					break
			prev_loss = loss
			pbar.set_description("iter %d, loss: %.4f" % (it, loss))
			self.num_iters += 1
		return tau, (pi, mu, sigma)

def cluster_pixels_gmm(image, K, max_iters=10, full_matrix=True):
	"""	
	Clusters pixels in the input image
	
	Each pixel can be considered as a separate data point (of length 3),
	which you can then cluster using GMM. Then, process the outputs into
	the shape of the original image, where each pixel is its most likely value.
	
	Args:
		image: input image of shape(H, W, 3)
		K: number of components
		max_iters: maximum number of iterations in GMM. Default is 10
		full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
	Return:
		clustered_img: image of shape(H, W, 3) after pixel clustering
	
	Hints:
		What do mu and tau represent?
	"""
	(H, W, _) = image.shape
	reshaped_image = image.reshape(-1, image.shape[-1]) # reshapes our 3D array into a 2D array shape = [(H * W), 3]
	gmm_image = GMM(reshaped_image, K, max_iters)
	tau, (_, mu, _)= gmm_image.__call__(full_matrix=full_matrix)
	
	max_values_indices = np.argmax(tau, axis=1) # highest proportion indicates which cluster pixel_i belongs to -> this finds which indices those max values are located
	mu_pixel_values = mu[max_values_indices] # 2D array, shape = [(H * W), 3] that contains average values of the image for each pixel
	clustered_img = mu_pixel_values.reshape(H, W, 3)
	
	return clustered_img
	
def density(points, pi, mu, sigma, gmm):
	"""	
	Evaluate the density at each point on the grid.
	Args:
		points: (N, 2) numpy array containing the coordinates of the points that make up the grid.
		pi: (K,) numpy array containing the mixture coefficients for each class
		mu: (K, D) numpy array containing the means of each cluster
		sigma: (K, D, D) numpy array containing the covariance matrixes of each cluster
		gmm: an instance of the GMM model
	
	Return:
		densities: (N, ) numpy array containing densities at each point on the grid
	
	HINT: You should be using the formula given in the hints.
	"""
	N, _ = points.shape
	K = len(pi) 
	densities = np.zeros(N)
	
	for k in range(K):
		# Multivariate normal density for each component
		normal_pdf = pi[k] * gmm.multinormalPDF(points, mu_i=mu[k], sigma_i=sigma[k]) 
		densities += normal_pdf
	
	return densities


def rejection_sample(xmin, xmax, ymin, ymax, pi, mu, sigma, gmm, dmax=1, M=0.1
	):
	"""	
	Performs rejection sampling. Keep sampling datapoints until d <= f(x, y) / M
	Args:
		xmin: lower bound on x values
		xmax: upper bound on x values
		ymin: lower bound on y values
		ymax: upper bound on y values
		gmm: an instance of the GMM model
		dmax: the upper bound on d
		M: scale_factor. can be used to control the fraction of samples that are rejected
	
	Return:
		x, y: the coordinates of the sampled datapoint
	
	HINT: Refer to the links in the hints
	"""
	while True:
		# randomly sample a point in the bounding box
		x = np.random.uniform(xmin, xmax)
		y = np.random.uniform(ymin, ymax)
		point = np.array([[x, y]]) 
		
		# compute density at this point
		density_value = density(point, pi, mu, sigma, gmm)[0]

		# sample a random value to decide acceptance
		d = np.random.uniform(0, dmax)

		# accept the point if d less than or equal to density_value / M
		if d <= density_value / M:
			return x, y
