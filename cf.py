import numpy as np
import pandas as pd
from scipy import sparse

def proc_col(col):
    """Encodes a pandas column with values between 0 and n-1.
 
    where n = number of unique values
    """
    uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx[x] for x in col]), len(uniq)

def encode_data(df):
    """Encodes rating data with continous user and movie ids using 
    the helpful fast.ai function from above.
    
    Arguments:
      train_csv: a csv file with columns user_id,movie_id,rating 
    
    Returns:
      df: a dataframe with the encode data
      num_users
      num_movies
      
    """
    ### BEGIN SOLUTION
    new_df = df.apply(proc_col)
    num_users=new_df.iloc[2][0]
    num_movies=new_df.iloc[2][1]
    result=pd.DataFrame(np.vstack((new_df.iloc[1][0], new_df.iloc[1][1], df['rating'].values)).T, columns=["userId","movieId","rating"])
    
    ### END SOLUTION
    return result, num_users, num_movies

def encode_new_data(df_val, df_train):
    """ Encodes df_val with the same encoding as df_train.
    Returns:
    df_val: dataframe with the same encoding as df_train
    """
    ### BEGIN SOLUTION
    df_train_encode=df_train.apply(proc_col)
    result=pd.DataFrame()
    #n(ested_dict = dict(df_train_encode.iloc[0][:2])
    result['userId']=df_val['userId'].map(df_train_encode.iloc[0][0])
    result['movieId']=df_val['movieId'].map(df_train_encode.iloc[0][1])
    result['rating'] = df_val['rating']
    result=result.dropna()
    result['userId']=result['userId'].astype(int)
    result['movieId']=result['movieId'].astype(int)

    
    ### END SOLUTION
    return result



def create_embedings(n, K):
    """ Create a numpy random matrix of shape n, K
    
    The random matrix should be initialized with uniform values in (0, 6/K)
    Arguments:
    
    Inputs:
    n: number of items/users
    K: number of factors in the embeding 
    
    Returns:
    emb: numpy array of shape (n, num_factors)
    """
    np.random.seed(3)
    emb = 6*np.random.random((n, K)) / K
    return emb


def df2matrix(df, nrows, ncols, column_name="rating"):
    """ Returns a sparse matrix constructed from a dataframe
    
    This code assumes the df has columns: MovieID,UserID,Rating
    """
    values = df[column_name].values
    ind_movie = df['movieId'].values
    ind_user = df['userId'].values
    return sparse.csc_matrix((values,(ind_user, ind_movie)),shape=(nrows, ncols))

def sparse_multiply(df, emb_user, emb_movie):
    """ This function returns U*V^T element wise multi by R as a sparse matrix.
    
    It avoids creating the dense matrix U*V^T
    """
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    
    df["Prediction"] = np.sum(emb_user[df["userId"].values]*emb_movie[df["movieId"].values], axis=1)
    return df2matrix(df, emb_user.shape[0], emb_movie.shape[0], column_name="Prediction")

def cost(df, emb_user, emb_movie):
    """ Computes mean square error

    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]

    Arguments:
      df: dataframe with all data or a subset of the data
      emb_user: embedings for users
      emb_movie: embedings for movies

    Returns:
      error(float): this is the MSE
    """
    ### BEGIN SOLUTION
    Y= df2matrix(df,emb_user.shape[0],emb_movie.shape[0])
    idx=Y.nonzero()

    prediction_matrix=sparse_multiply(df,emb_user,emb_movie)
    error=np.sum((np.power(Y[idx]-prediction_matrix[idx],2)))/Y.count_nonzero()

    # encoded_df, num_users, num_movies = encode_data(df)
    # actual = df2matrix(encoded_df, num_users, num_movies)
    # predictions = sparse_multiply(encoded_df, emb_user, emb_movie)
    # non_zero_idx = actual.nonzero()
    # error = np.mean(np.power(actual[non_zero_idx] - predictions[non_zero_idx], 2))

    ### END SOLUTION
    return error

def finite_difference(df, emb_user, emb_movie, ind_u=None, ind_m=None, k=None):
    """ Computes finite difference on MSE(U, V).
    
    This function is used for testing the gradient function. 
    """
    e = 0.000000001
    c1 = cost(df, emb_user, emb_movie)
    K = emb_user.shape[1]
    x = np.zeros_like(emb_user)
    y = np.zeros_like(emb_movie)
    if ind_u is not None:
        x[ind_u][k] = e
    else:
        y[ind_m][k] = e
    c2 = cost(df, emb_user + x, emb_movie + y)
    return (c2 - c1)/e

def gradient(df, Y, emb_user, emb_movie):
    """ Computes the gradient.
    
    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]
    
    Arguments:
      df: dataframe with all data or a subset of the data
      Y: sparse representation of df
      emb_user: embedings for users
      emb_movie: embedings for movies
      
    Returns:
      d_emb_user
      d_emb_movie
    """
    ### BEGIN SOLUTION
    prediction_matrix=sparse_multiply(df,emb_user,emb_movie)
    delta=(Y-prediction_matrix)
    N=len(df)

    grad_user= -2/N * (delta*(emb_movie))
    grad_movie= -2/N * (delta.T*(emb_user))

    ### END SOLUTION
    return grad_user, grad_movie

# you can use a for loop to iterate through gradient descent
def gradient_descent(df, emb_user, emb_movie, iterations=100, learning_rate=0.01, df_val=None):
    """ Computes gradient descent with momentum (0.9) for a number of iterations.
    
    Prints training cost and validation cost (if df_val is not None) every 50 iterations.
    
    Returns:
    emb_user: the trained user embedding
    emb_movie: the trained movie embedding
    """
    Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
    ### BEGIN SOLUTION
    beta=0.9
    v_user=0
    v_movie=0

    for i in range(iterations):
        grad_user, grad_movie = gradient(df, Y, emb_user, emb_movie)

        v_user=beta*v_user+(1-beta)*grad_user
        emb_user=emb_user-learning_rate*v_user

        v_movie=beta*v_movie+(1-beta)*grad_movie
        emb_movie=emb_movie-learning_rate*v_movie

        if((i+1)%50==0):
            print(cost(df,emb_user,emb_movie))
            if(df_val is not None):
                print(cost(df_val,emb_user,emb_movie))

    ### END SOLUTION
    return emb_user, emb_movie

