import pandas as pd
import ydf
from matplotlib import pyplot as plt




#training data
df = pd.read_csv('train.csv')
#unclassified data
df_test = pd.read_csv('test.csv')



#preprocessing and feature engineering for training data
#turn embarked and sex into numeric variables
df['Embarked'] = df['Embarked'].replace(  ['S', 'C', 'Q'] , [0,1,2] )
df['Embarked'] = df['Embarked'].fillna(3)
df['Embarked']=df['Embarked'].astype(int)
df['Sex'] = df['Sex'].replace( ['male', 'female'], [0,1] )

#age: replace nan with average age grouped by boys (masters), girls (miss), men (mr.), women (mrs.), doctors (dr.)
AgeBoys = df [df['Name'].str.contains('Master')] ['Age'].mean()
df [df['Name'].str.contains('Master')] = df [df['Name'].str.contains('Master')].fillna(AgeBoys)
AgeGirls = df [df['Name'].str.contains('Miss.')] ['Age'].mean()
df [df['Name'].str.contains('Miss.')] = df [df['Name'].str.contains('Miss.')].fillna(AgeGirls)
AgeMen = df [df['Name'].str.contains('Mr.')] ['Age'].mean()
df [df['Name'].str.contains('Mr.')] = df [df['Name'].str.contains('Mr.')].fillna(AgeMen)
AgeWomen = df [df['Name'].str.contains('Mrs.')] ['Age'].mean()
df [df['Name'].str.contains('Mrs.')] = df [df['Name'].str.contains('Mrs.')].fillna(AgeWomen)
AgeDr = df [df['Name'].str.contains('Dr.')] ['Age'].mean()
df [df['Name'].str.contains('Dr.')] = df [df['Name'].str.contains('Dr.')].fillna(AgeDr)

#create variable last name
df['LastName']=df['Name'].str.split(',').str[0]

#create variable for 'all family members survived' (-1:no ,  1:yes)
df = df.merge( df.groupby('LastName')['Survived'].mean() , how='right', on='LastName')
df['FamSurvived'] = df['Survived_y']==1
df['FamSurvived'] = df['FamSurvived'].astype(int)
df['FamSurvived'] = df['FamSurvived'].replace(0,-1)
#create variable family size
df['FamSize'] = df['SibSp'] + df['Parch']
# df['FamSurvRating'] = np.sqrt( df['FamSurvived'] * df['FamSize'] )

#create variable 'HasFamily'
df['HasFamily']=df['FamSize'] > 0

#adapt 'all family members survived' : if passenger actually has no family, set to 0 (even if passenger survived)
df['FamSurvived'] = df['FamSurvived']*df['HasFamily']

#create target variable
Y=df['Survived_x']



#preprocess test set

#turn embarked and sex into numeric variables
df_test['Embarked'] = df_test['Embarked'].replace(  ['S', 'C', 'Q'] , [0,1,2] )
df_test['Embarked'] = df_test['Embarked'].fillna(3)
df_test['Embarked']=df_test['Embarked'].astype(int)
df_test['Sex'] = df_test['Sex'].replace( ['male', 'female'], [0,1] )

#replace nan with average age grouped by boys (masters), girls (miss), men (mr.), women (mrs.), doctors (dr.)
AgeBoys = df_test [df_test['Name'].str.contains('Master')] ['Age'].mean()
df_test [df_test['Name'].str.contains('Master')] = df_test [df_test['Name'].str.contains('Master')].fillna(AgeBoys)
AgeGirls = df_test [df_test['Name'].str.contains('Miss.')] ['Age'].mean()
df_test [df_test['Name'].str.contains('Miss.')] = df_test [df_test['Name'].str.contains('Miss.')].fillna(AgeGirls)
AgeMen = df_test [df_test['Name'].str.contains('Mr.')] ['Age'].mean()
df_test [df_test['Name'].str.contains('Mr.')] = df_test [df_test['Name'].str.contains('Mr.')].fillna(AgeMen)
AgeWomen = df_test [df_test['Name'].str.contains('Mrs.')] ['Age'].mean()
df_test [df_test['Name'].str.contains('Mrs.')] = df_test [df_test['Name'].str.contains('Mrs.')].fillna(AgeWomen)
AgeDr = df_test [df_test['Name'].str.contains('Dr.')] ['Age'].mean()
df_test [df_test['Name'].str.contains('Dr.')] = df_test [df_test['Name'].str.contains('Dr.')].fillna(AgeDr)

#create last name variable
df_test['LastName']=df_test['Name'].str.split(',').str[0]

#create FamSize variable
df_test['FamSize'] = df_test['SibSp'] + df_test['Parch']

#create FamSurvived variable for test data
    #1: person has family and whole family survived
    #0: person has no family in train set
    #-1: person has family and not whole family survived
df_test['HasFamily']=df_test['SibSp'] + df_test['Parch'] > 0
toMap=df[['LastName', 'FamSurvived']].drop_duplicates(subset=['LastName'])
toMap=toMap.set_index('LastName')

#map FamSurvived from test set via last name. If passenger has no family and / or there is no data on the family in training set, set FamSurvived to 0
df_test['FamSurvived'] = df_test['LastName'].map(toMap['FamSurvived'])
df_test['FamSurvived']=df_test['FamSurvived'] * df_test['HasFamily']
df_test['FamSurvived']=df_test['FamSurvived'].fillna(0)


#create train set X: drop variables that are not used for training

df_Surv = df['Survived_x']
X = df.drop(['PassengerId', 'Name', 'LastName', 'Survived_x', 'Survived_y', 'Ticket', 'Cabin', 'SibSp','Parch'], axis=1)

#normalize data
X = (X-X.mean())/X.std()


#create test set Z: drop variables that are not used for model
Z = df_test.drop(['Name', 'LastName', 'PassengerId' , 'Ticket', 'Cabin' , 'SibSp', 'Parch'], axis=1)
#normalize data
Z = (Z-Z.mean())/Z.std()
#sort columns of Z by columns of X
Z=Z[X.columns]

#create label variable
X['Survived']=df_Surv


#ydf model

tuner= ydf.RandomSearchTuner(num_trials=250)
tuner.choice('num_trees', range(200,400))
tuner.choice('max_depth', range(1,25))


learner=ydf.RandomForestLearner(
    label="Survived",
    # split_axis="SPARSE_OBLIQUE",
    # sparse_oblique_normalization="MIN_MAX",
    # sparse_oblique_num_projections_exponent=1.5,
    tuner=tuner,
)


eval=learner.cross_validation(X, folds=10)
print(eval)

model=learner.train(X)

predictions=model.predict(Z)
predictions=[0 if val < 0.5 else 1 for val in predictions]

model.describe()
model.analyze(X)



# from titanic import NN_predictions

# plt.plot(NN_predictions)
plt.hist(predictions)
plt.legend(['Survived'])
plt.show()


#create and save csv file for submission
submission =  pd.DataFrame( df_test['PassengerId'] )
submission['Survived'] = predictions
submission.to_csv('submission.csv', index=False)


