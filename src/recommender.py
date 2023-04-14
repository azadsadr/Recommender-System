import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

def user_with_no_history(utility, user_id):
    try:
        if utility[user_id, :].count_nonzero() > 0:
            return False
        else:
            return True
    except:
        return True
    

def RecSys(model, U, movies, user_id, num_recomm):

    if not isinstance(U, (csr_matrix)):
        raise Exception("scipy sparse matrix expected!")
    
    if user_id not in range(U.shape[0]):
        raise Exception("user id is out of range!")
    
    # row and column indices for unrated values for specified user in utility matrix
    _, cols = np.where(U[user_id, :].toarray() == 0)

    users = torch.LongTensor(len(cols) * [user_id]) #.cuda()
    items = torch.LongTensor(cols) #.cuda()

    model.eval()
    output = model(users, items)
    sorted_output = torch.sort(output.detach(), descending=True)
    idx = np.array(sorted_output.indices[:num_recomm]).tolist()

    res = movies[movies['movieId'].isin(idx)]
    return res
    
def recommend(model, ratings, movies, stats, user_id, num_recom):

    if not isinstance(ratings, (pd.DataFrame)):
        raise Exception("ratings argument is not dataframe!")
    
    num_users = ratings.userId.nunique()    # no. of unique users
    num_items = ratings.movieId.nunique()   # no. of unique movies
    rows = ratings.userId.values            # user id's
    cols = ratings.movieId.values           # movie id's
    rat = ratings.rating.values             # ratings
    utility = csr_matrix((rat, (rows, cols)), shape=(num_users, num_items))
    
    if not isinstance(user_id, (int)) or user_id < 0:
        raise Exception("invalid user id!")

    if user_with_no_history(utility, user_id):
        return stats.iloc[0:num_recom,:]
    else:
        return RecSys(model, utility, movies, user_id, num_recom)