import numpy as np
from numpy.linalg import inv
from numpy import transpose
'''
To evaluate your policy you can use (policy.py and runner.py). Your task is to complete the functions recommend, update and set_articles in the policy file. We will first call the set_articles method and pass in all the article features. Then for every line in in webscope-logs.txt the provided runner will call your recommend function. If your chosen article matches the displayed article in the log, the result of choosing this article (click or no click) is fed back to your policy and you have a chance to update the current model accordingly. This is achieved via the update method.
'''

np.random.seed(1)
#inp = raw_input().split()
#alpha = float(inp[0])
#R0 = float(inp[1])
#R1 = float(inp[2])
alpha = 1
R0 = -5
R1 = 25

d = 6

article_features        = None
A                       = {}
A_inverse               = {}
b                       = {}
theta                   = {}
cc                      = None
current_user            = None

def set_articles(articles):

    global article_features

    article_features = [art for art in articles]

    for article in articles:
        A[article] = np.identity(d)
        A_inverse[article] = np.identity(d)
        b[article] = np.zeros((d,1))
        theta[article] = np.zeros((d, 1))


def update(reward):
    if reward == -1 :
        return
    if reward == 1 :
        r = R1
    else :
        r = R0

    A[cc] += np.dot(current_user, np.transpose(current_user))
    A_inverse[cc] = np.linalg.inv(A[cc])
    b[cc] += r * current_user
    theta[cc] = A_inverse[cc].dot(b[cc])


def recommend(time, user_features, choices) :
    article_max_id = None
    maxUCB = -1

    user = np.asarray(user_features)
    user.shape = (d, 1)

    for art in choices :
        if art not in A :
            A[art] = np.identity(d)
            A_inverse[art] = np.identity(d)
            b[art] = np.zeros((d, 1))
            theta[art] = np.zeros((d, 1))

            article_max_id = art
            break
        else :
            theta_art_transpose = np.transpose(theta[art])
            theta_art_transpose_x = np.dot(theta_art_transpose, user)
            user_transpose = np.transpose(user)
            user_transpose_A_inverse_art = np.dot(user_transpose, A_inverse[art])
            user_transpose_A_inverse_art_user = np.dot(user_transpose_A_inverse_art, user)
            s = alpha * np.sqrt(user_transpose_A_inverse_art_user)

            UCB = theta_art_transpose_x + s

            if UCB > maxUCB :
                article_max_id = art
                maxUCB = UCB
    
    global cc 
    cc = article_max_id
    global current_user
    current_user = user

    return article_max_id

