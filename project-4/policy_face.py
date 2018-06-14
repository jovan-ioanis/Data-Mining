import numpy as np

alpha = 1.1 

d = 6

r1 = 12 

r0 = -1.5

# (article_id, matrix M as in LinUCB)
M = dict() 
# (article_id, inverse of M)
M_inv = dict() 
# (article_id, b as in LinUCB)
b = dict()  
# (article_id, weights as in LinUCB)
w = dict()  

article_list = None

# Remember so we know what to update
# the article id of recommended article
last_article_id = None
# user features in last recommendation
last_user_features = None

def set_articles(articles):
    global article_list

    # list of article's ids 
    article_list = [x for x in articles]       

    # initialize all the variables
    for article_id in article_list:
        M[article_id] = np.identity(d)
        M_inv[article_id] = np.identity(d)
        b[article_id] = np.zeros((d, 1))
        w[article_id] = np.zeros((d, 1))


def update(reward):
   
    if reward == -1:   
        return
    
    # having different scaling parameters for 
    # cases reward = 0,1 yields much better score
    if reward == 1:
        r = r1
    else:
        r = r0

    # update as in linUCB
    M[last_article_id] += last_user_features.dot(last_user_features.T)
    M_inv[last_article_id] = np.linalg.inv(M[last_article_id])
    b[last_article_id] += r * last_user_features
    w[last_article_id] = M_inv[last_article_id].dot(b[last_article_id])   


def recommend(time, user_features, articles):
    best_article_id = None
    best_ucb_value = -1

    user_features = np.asarray(user_features)

    user_features.shape = (d, 1)

    for article_id in articles:
 
        # if this is the first time we meet this article
        if article_id not in M:
            # initialize the variables for current articles
            # that has not been seen before
            M[article_id] = np.identity(d)
            M_inv[article_id] = np.identity(d)
            b[article_id] = np.zeros((d, 1))
            w[article_id] = np.zeros((d, 1))

            # Get at least 1 datapoint for this article
            best_article_id = article_id
            break

        # otherwise, we have already seen this article
        else:
            ucb_value = w[article_id].T.dot(user_features) +\
                        alpha * np.sqrt(user_features.T.dot(M_inv[article_id]).dot(user_features))

            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value
                best_article_id = article_id

    global last_article_id


    # remember the article that we want to recommend
    last_article_id = best_article_id  
    global last_user_features

    # remember user features
    last_user_features = user_features 

    return best_article_id