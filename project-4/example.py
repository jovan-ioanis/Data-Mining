import numpy as np
#from nimpy import linalg

chosen_article = -1
chosen_id = -1

R1 = 20
R0 = -9

#delta = 0.3
alpha = 1.2 #+ np.sqrt(np.log(2/delta)/2)

article_features = {}										# article features
d = 5														# dimension of user features
a = 2
k = a*d														# dimension of article features
A0 = np.identity(k)											# identity matrix A0 						kxk
A0_inverse = np.identity(k)									# inverse identity matrix A0^-1 			kxk
b0 = np.zeros((k,1))											# zero vector

# DISJOINT PART:
Aa = {}														# matrix per article a 						6x6
Aa_inverse = {}												# inverse matrix per article a 				6x6
ba = {}														# vector per article a
# HYBRID PART:
Ba = {}														# matrix per article a 						6xk
Ba_transpose = {}											# transposed matrix per article a 			kx6

# TO SPEED UP CALCULATIONS
Aa_inverse_ba = {}
Aa_inverse_Ba = {}
Ba_transpose_Aa_inverse = {}
theta = {}
beta = np.zeros((k, 1))
index2id = {}
ARTICLE_COUNT = 0

z = None
z_transpose = None
x = None
x_transpose = None


def init() :

	global chosen_article
	global article_features
	global A0
	global A0_inverse
	global b0
	global Aa
	global Aa_inverse
	global ba
	global Ba
	global Ba_transpose
	global Aa_inverse_ba
	global Aa_inverse_Ba
	global Ba_transpose_Aa_inverse
	global theta
	global beta
	global index2id
	global ARTICLE_COUNT
	global z
	global z_transpose
	global x
	global x_transpose
	global alpha

	A0 = 						np.identity(k)
	A0_inverse = 				np.identity(k)
	b0 = 						np.zeros((k, 1))

	article_features = 			np.zeros((ARTICLE_COUNT, 1, a))
	Aa = 						np.zeros((ARTICLE_COUNT, d, d))
	Aa_inverse = 				np.zeros((ARTICLE_COUNT, d, d))
	Ba = 						np.zeros((ARTICLE_COUNT, d, k))
	Ba_transpose = 				np.zeros((ARTICLE_COUNT, k, d))
	ba = 						np.zeros((ARTICLE_COUNT, d, 1))

	Aa_inverse_ba = 			np.zeros((ARTICLE_COUNT, d, 1))
	Aa_inverse_Ba = 			np.zeros((ARTICLE_COUNT, d, k))
	Ba_transpose_Aa_inverse = 	np.zeros((ARTICLE_COUNT, k, d))
	theta = 					np.zeros((ARTICLE_COUNT, d, 1))

def set_articles(articles):
	global chosen_article
	global article_features
	global A0
	global A0_inverse
	global b0
	global Aa
	global Aa_inverse
	global ba
	global Ba
	global Ba_transpose
	global Aa_inverse_ba
	global Aa_inverse_Ba
	global Ba_transpose_Aa_inverse
	global theta
	global beta
	global index2id
	global ARTICLE_COUNT
	global z
	global z_transpose
	global x
	global x_transpose
	global alpha

	ARTICLE_COUNT = len(articles)

	init()

	index = 0
	for key in articles.keys() :
		print(articles[key])
		index2id[key] = 					index
		article_features[index] = 			articles[key][0:2]
		Aa[index] = 						np.identity(d)
		Aa_inverse[index] = 				np.identity(d)
		Ba[index] = 						np.zeros((d, k))
		Ba_transpose[index] = 				np.zeros((k, d))
		ba[index] = 						np.zeros((d, 1))
		Aa_inverse_ba[index] = 				np.zeros((d, 1))
		Aa_inverse_Ba[index] = 				np.zeros((d, k))
		Ba_transpose_Aa_inverse[index] = 	np.zeros((k, d))
		theta[index] = 						np.zeros((d, 1))
		index += 1


def update(reward) :
	global R1
	global R0

	r = 0

	if reward == -1 :
		return
	if reward == 0 :
		r = R0
	if reward == 1 :
		r = R1

	#print("==== UPDATE ====")
	#print(r)
	global chosen_article
	global article_features
	global A0
	global A0_inverse
	global b0
	global Aa
	global Aa_inverse
	global ba
	global Ba
	global Ba_transpose
	global Aa_inverse_ba
	global Aa_inverse_Ba
	global Ba_transpose_Aa_inverse
	global theta
	global beta
	global index2id
	global ARTICLE_COUNT
	global z
	global z_transpose
	global x
	global x_transpose
	global alpha

	# print("A0 shape is " + str(A0.shape))

	A0 += np.dot(Ba_transpose_Aa_inverse[chosen_article], Ba[chosen_article])
	b0 += np.dot(Ba_transpose_Aa_inverse[chosen_article], ba[chosen_article])

	Aa[chosen_article] += np.dot(x, x_transpose)
	Aa_inverse[chosen_article] = np.linalg.inv(Aa[chosen_article])

	Ba[chosen_article] += np.dot(x, z_transpose)
	Ba_transpose[chosen_article] = np.transpose(Ba[chosen_article])

	ba[chosen_article] += r * x 
	Aa_inverse_ba[chosen_article] = np.dot(Aa_inverse[chosen_article], ba[chosen_article])
	Aa_inverse_Ba[chosen_article] = np.dot(Aa_inverse[chosen_article], Ba[chosen_article])
	Ba_transpose_Aa_inverse[chosen_article] = np.dot(Ba_transpose[chosen_article], Aa_inverse[chosen_article])	
	
	
	#print("Chosen article is " + str(chosen_article) + " and Ba_transpose_Aa_inverse shape is " + str(Ba_transpose_Aa_inverse.shape))
	A0 += np.dot(z, z_transpose) - np.dot(Ba_transpose_Aa_inverse[chosen_article], Ba[chosen_article])
	b0 += r*z - np.dot(Ba_transpose[chosen_article], np.dot(Aa_inverse[chosen_article], ba[chosen_article]))	

	A0_inverse = np.linalg.inv(A0)
	beta = np.dot(A0_inverse, b0)
	#print("beta shape is " + str(beta.shape))
	#print("Aa_inverse_ba[chosen_article] shape is " + str(Aa_inverse_ba[chosen_article].shape))
	#print("Aa_inverse_Ba[chosen_article] shape is " + str(Aa_inverse_Ba[chosen_article].shape))
	Aa_inverse_Ba_chosen_beta = np.dot(Aa_inverse_Ba[chosen_article], beta)
	#print("Aa_inverse_Ba_chosen_beta shape is " + str(Aa_inverse_Ba_chosen_beta.shape))
	theta[chosen_article] = Aa_inverse_ba[chosen_article] - Aa_inverse_Ba_chosen_beta



def recommend(time, user_features, choices) :
	global chosen_article
	global article_features
	global A0
	global A0_inverse
	global b0
	global Aa
	global Aa_inverse
	global ba
	global Ba
	global Ba_transpose
	global Aa_inverse_ba
	global Aa_inverse_Ba
	global Ba_transpose_Aa_inverse
	global theta
	global beta
	global index2id
	global ARTICLE_COUNT
	global z
	global z_transpose
	global x
	global x_transpose
	global alpha
	global chosen_id

	#print(Aa_inverse)

	choices_count = len(choices)
	x_transpose = np.array([user_features[1:]])
	#print("x_transpose len is " + str(x_transpose.shape))
	x = np.transpose(x_transpose)
	#print("x len is " + str(x.shape))

	index = [index2id[article] for article in choices]

	article_features_chosen = np.array([article_features[key] for key in index])
	#print("article_features_chosen shape is " + str(article_features_chosen.shape))

	# article_features_chosen = article_features[index]

	z_transpose_chosen = np.einsum('i,j', article_features_chosen.reshape(-1), user_features[1:]).reshape(choices_count, 1, k)
	#print("z_transpose_chosen shape is " + str(z_transpose_chosen.shape))
	z_chosen = np.transpose(z_transpose_chosen, (0, 2, 1))
	#print("z_chosen shape is " + str(z_chosen.shape))

	Ba_transpose_Aa_inverse_chosen = np.array([Ba_transpose_Aa_inverse[key] for key in index])

	#print("Ba_transpose_Aa_inverse_chosen shape is " + str(Ba_transpose_Aa_inverse_chosen.shape))

	Ba_transpose_Aa_inverse_chosen_x = np.dot(Ba_transpose_Aa_inverse_chosen, x)
	#print("Ba_transpose_Aa_inverse_chosen_x shape is " + str(Ba_transpose_Aa_inverse_chosen_x.shape))
	Ba_transpose_Aa_inverse_chosen_x_transpose = np.transpose(Ba_transpose_Aa_inverse_chosen_x, (0, 2, 1))
	#print("Ba_transpose_Aa_inverse_chosen_x_transpose shape is " + str(Ba_transpose_Aa_inverse_chosen_x_transpose.shape))
	Ba_transpose_Aa_inverse_chosen_x_transpose_A0_inverse_transpose = np.dot(Ba_transpose_Aa_inverse_chosen_x_transpose, np.transpose(A0_inverse))
	#print("Ba_transpose_Aa_inverse_chosen_x_transpose_A0_inverse_transpose shape is " + str(Ba_transpose_Aa_inverse_chosen_x_transpose_A0_inverse_transpose.shape))

	A0_inverse_Ba_transpose_Aa_inverse_x_chosen = np.transpose(Ba_transpose_Aa_inverse_chosen_x_transpose_A0_inverse_transpose, (0, 2, 1))
	#print("A0_inverse_Ba_transpose_Aa_inverse_x_chosen shape is " + str(A0_inverse_Ba_transpose_Aa_inverse_x_chosen.shape))

	A0_inverse_z_chosen = np.transpose(np.dot(z_transpose_chosen, np.transpose(A0_inverse)), (0, 2, 1))
	#print("A0_inverse_z_chosen shape is " + str(A0_inverse_z_chosen.shape))

	diff = A0_inverse_z_chosen - 2 * A0_inverse_Ba_transpose_Aa_inverse_x_chosen
	#print("diff shape is " + str(diff.shape))

	first_term = np.sum(z_chosen.reshape(choices_count, k, 1, 1) * diff.reshape(choices_count, k, 1, 1), -3)
	#print("first_term shape is " + str(first_term.shape))

	Aa_inverse_chosen = np.array([Aa_inverse[key] for key in index])
	#print("Aa_inverse_chosen shape is " + str(Aa_inverse_chosen.shape))
	Aa_inverse_Ba_chosen = np.array([Aa_inverse_Ba[key] for key in index])
	#print("Aa_inverse_Ba_chosen shape is " + str(Aa_inverse_Ba_chosen.shape))

	Aa_inverse_chosen_x = np.dot(Aa_inverse_chosen, x)
	#print("Aa_inverse_chosen_x shape is " + str(Aa_inverse_chosen_x.shape))
	Aa_inverse_Ba_A0_inverse_Ba_transpose_Aa_inverse_x_chosen = 														\
		np.sum(np.transpose(Aa_inverse_Ba_chosen, (0, 2, 1)).reshape(choices_count, k, d, 1) * 							\
							A0_inverse_Ba_transpose_Aa_inverse_x_chosen.reshape(choices_count, k, 1, 1), -3)
	#print("Aa_inverse_Ba_A0_inverse_Ba_transpose_Aa_inverse_x_chosen shape is " + str(Aa_inverse_Ba_A0_inverse_Ba_transpose_Aa_inverse_x_chosen.shape))

	summm = Aa_inverse_chosen_x + Aa_inverse_Ba_A0_inverse_Ba_transpose_Aa_inverse_x_chosen
	#print("summ shape is " + str(summm.shape))

	second_term = np.transpose(np.dot(np.transpose(summm, (0, 2, 1)), x), (0, 2, 1))
	#print("second_term shape is " + str(second_term.shape))

	Sa = first_term + second_term
	#print("Sa shape is " + str(Sa.shape))

	theta_chosen = np.array([theta[key] for key in index])
	#print("theta_chosen shape is " + str(theta_chosen.shape))

	x_transpose_theta_chosen = np.transpose(np.dot(np.transpose(theta_chosen, (0, 2, 1)), x), (0, 2, 1))
	#print("x_transpose_theta_chosen shape is " + str(x_transpose_theta_chosen.shape))
	z_transpose_beta_chosen = np.dot(z_transpose_chosen, beta)
	#print("z_transpose_beta_chosen shape is " + str(z_transpose_beta_chosen.shape))
	alpha_sqrt_Sa = alpha * np.sqrt(Sa)
	#print("Alpha  = " + str(alpha))
	#print("alpha_sqrt_Sa shape is " + str(alpha_sqrt_Sa.shape))

	article_with_highest_UCB = np.argmax(z_transpose_beta_chosen + x_transpose_theta_chosen + alpha_sqrt_Sa)

	z = z_chosen[article_with_highest_UCB]
	z_transpose = z_transpose_chosen[article_with_highest_UCB]
	chosen_article = index[article_with_highest_UCB]
	#print("Highest UCB for article " + str(article_with_highest_UCB) + " and its ID is " + str(choices[article_with_highest_UCB]))
	chosen_id = choices[article_with_highest_UCB]

	return choices[article_with_highest_UCB]
        