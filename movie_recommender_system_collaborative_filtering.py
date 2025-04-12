import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from implicit.als import AlternatingLeastSquares
import numpy as np
import seaborn as sns
from sklearn.model_selection import ParameterGrid, train_test_split
import optuna

def get_top_n_recommendations(user_id,user_factors,item_factors,n=5):
    user_vector=user_factors[user_id]
    scores=np.dot(user_vector,item_factors.T)
    minScore=scores.min()
    maxScore=scores.max()
    scaled_scores=(scores-minScore)/(maxScore-minScore)
    top_items=np.argsort(scaled_scores)[::-1][:n]
    return [(item,scaled_scores[item]) for item in top_items]

def evaluate_model(user_factors,item_factors,test_data,train_data,n=5):
    total_error=0
    train_user_ids=train_data['UserID'].unique()
    test_user_ids=test_data['UserID'].unique()
    common_user_ids=np.intersect1d(train_user_ids,test_user_ids)
    for user_id in common_user_ids[:50]:
        actual_ratings=test_data[test_data['UserID']==user_id].sort_values(by='Rating',ascending=False)
        top_recommendations=get_top_n_recommendations(user_id,user_factors,item_factors,n=n)
        for item,scaled_Score in top_recommendations:
            actual_rating=actual_ratings[actual_ratings['MovieID']==item]['Rating'].values
            if len(actual_rating)>0:
                error=(scaled_Score-actual_rating[0])**2
                total_error+=error
    mse=total_error/len(common_user_ids)
    return -mse


def objective(trial):
    global best_score,best_model
    best_model=None
    best_score=-10000000000

    factors=trial.suggest_int('factors',10,60)
    regularization=trial.suggest_float('regularization',0.00001,0.1,log=True)
    iterations=trial.suggest_int('iterations',10,50)

    # train_data,test_data=train_test_split(df_final,test_size=0.2,random_state=42)
    model=AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=42,
        calculate_training_loss=True,
        use_gpu=False
    )
    
    model.fit(user_item_csr)

    user_factors,item_factors=model.user_factors,model.item_factors

    score=evaluate_model(user_factors,item_factors,test_data,train_data)
    
    if score>best_score :
        best_score=score
        best_model=model
    return best_score

def user_evaluation(df,user_id_to_evaluate,test_data):
    global best_score,best_model
    actual_ratings=test_data[test_data['UserID']==user_id_to_evaluate]

    actual_ratings=actual_ratings.sort_values(by='Rating',ascending=False)

    print(f"\n Top 5 rated movies by user {user_id_to_evaluate}:")
    for i,(_,row) in enumerate(actual_ratings.head(5).iterrows()):
        title=row['Title']
        genres=row['Genres']
        rating=row['Rating']
        print(f'{i+1}. movie: {title}, Genres: {genres}, Rating: {rating:.2f}')
    user_factors,item_factors=best_model.user_factors,best_model.item_factors
    top_recomm=get_top_n_recommendations(user_id_to_evaluate,user_factors,item_factors,n=5)

    moviedf=df_final[['MovieID','Genres','Title']].drop_duplicates()

    recomm_movie_info=moviedf[moviedf['MovieID'].isin([item[0] for item in top_recomm])]

    print(f"\n Top 5 recommendations for user {user_id_to_evaluate}:")
    for i,(_,row) in enumerate(recomm_movie_info.iterrows()):
        title=row['Title']
        genres=row['Genres']
        scaled_score=top_recomm[i][1]
        print(f"{i+1}. Movie:{title},genres:{genres},Recommendation score:{scaled_score:.2f}")

if __name__=='__main__':
    global best_score,best_model
    #read movies data
    df_movies=pd.read_csv('/Users/drago/Documents/Practicefiles/Data_files/MovieRecommenderSystem/ml-1m/movies.dat',delimiter='::',engine='python',header=None,names=['MovieID', 'Title', 'Genres'],encoding='ISO-8859-1')
    df_rating=pd.read_csv('/Users/drago/Documents/Practicefiles/Data_files/MovieRecommenderSystem/ml-1m/ratings.dat',delimiter='::',engine='python',header=None,names=['UserID', 'MovieID', 'Rating', 'Timestamp'],encoding='ISO-8859-1')
    df_users=pd.read_csv('/Users/drago/Documents/Practicefiles/Data_files/MovieRecommenderSystem/ml-1m/users.dat',delimiter='::',engine='python',header=None,names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],encoding='ISO-8859-1')
    
    df=df_rating.merge(df_movies,on='MovieID',how='left')
    df_final=df.merge(df_users,on='UserID',how='right')
    df_final['Timestamp']=pd.to_datetime(df_final['Timestamp'],unit='s')

    #dropping non essential columns
    df_final=df_final.drop(['Gender','Age','Occupation','Zip-code'],axis=1)
    df_final['Rating_ceil']=np.ceil(df['Rating'])

    movie_df=df_final[['MovieID','Title','Genres']].drop_duplicates()

    train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)

    sparse_user_item=pd.pivot_table(train_data,values='Rating',index='UserID',columns='MovieID').fillna(0)
    user_item_csr=csr_matrix(sparse_user_item)
    study=optuna.create_study()

    final=study.optimize(objective,n_trials=100)

    print(f'Best_model={best_model},best score={best_score},best parameters={study.best_trial.params}, final={final}')

    user_id_to_evaluate=1000 #input. depends on user
    user_evaluation(df_final,user_id_to_evaluate,test_data)
