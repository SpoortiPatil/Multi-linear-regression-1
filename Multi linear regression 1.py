# Importing the necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
strt= pd.read_csv("50_Startups.csv")
strt.head()
strt.columns

# Creating the dummies for discrete dataset of "State"
dummies= pd.get_dummies(strt['State'])
dummies

strt= pd.concat([strt,dummies], axis=1)        # Combining the dummies with the dataset
strt

strt= strt.drop("State",axis=1)
strt

# Renaming the columns for convinience
strt = strt.rename(columns= {"R&D Spend":"RD_spnd", "Administration":"Admin", "Marketing Spend":"Mark_sp", "New York":"New_york"})
strt.columns

# Correlation matrix
strt.corr()
# There exists some collinearity between input variables especially between RD_spnd and Mark_sp

import seaborn as sns
# Scatter plot between the variables along with histograms
sns.pairplot(strt)

strt.columns

import statsmodels.formula.api as smf
# preparing model considering all the variables

ml1= smf.ols("Profit~RD_spnd+Admin+Mark_sp+California+Florida+New_york", data=strt).fit()
ml1.params
ml1.summary()

# preparing model based only on Admin
ml_ad= smf.ols("Profit~Admin", data= strt).fit()
ml_ad.params
ml_ad.summary()

# preparing model based only on Market_spnd
ml_ms= smf.ols("Profit~ Mark_sp", data= strt).fit()
ml_ms.summary()

# preparing model based on Market_spnd and Admin
ml_ma= smf.ols("Profit~Mark_sp+Admin", data=strt).fit()
ml_ma.summary()


# Both coefficients p-value became significant, but with RD_spnd only one of the variable can be considered
# preparing model based on Market_spnd and RD_spnd
ml_rm= smf.ols("Profit~ Mark_sp+RD_spnd", data= strt).fit()
ml_rm.summary()

# Mark_spnd p-value became insignificant when it is used along with RD_spnd, the reason is high-collinearity between both
# preparing model based on Admin and RD_spnd
ml_ra= smf.ols("Profit~ Admin+RD_spnd", data= strt).fit()
ml_ra.summary()

ml_ram= smf.ols("Profit~Admin+RD_spnd+Mark_sp", data=strt).fit()
ml_ram.summary()

# Admin p-value became insignificant when it is used along with RD_spnd
import statsmodels.api as sm
# Checking whether data has any influential values
sm.graphics.influence_plot(ml1)            # influence index plots
# index 48 AND 49 is showing high influence so we can exclude that entire row
strt.new= strt.drop(strt.index[[48,49]], axis=0)
strt.new.head()

# Preparing new model
ml_new= smf.ols("Profit~Admin+RD_spnd+Mark_sp", data=strt.new).fit()
ml_new.summary()
ml_new.params                          # Getting coefficients of variables

print(ml_new.conf_int(0.01))

# predicted values of profit
pred= ml_new.predict(strt.new)
pred

# calculating VIF's values of independent variables
rsq_rd= smf.ols("RD_spnd~Admin+Mark_sp", data=strt.new).fit().rsquared
vif_rd= 1/(1-rsq_rd)
vif_rd

rsq_ad= smf.ols("Admin~RD_spnd+Mark_sp", data= strt.new).fit().rsquared
vif_ad= 1/(1-rsq_ad)
vif_ad

rsq_ms= smf.ols("Mark_sp~Admin+RD_spnd", data= strt.new).fit().rsquared
vif_ms= 1/(1-rsq_ms)
vif_ms

# Storing vif values in a data frame
d1= {"Variables": ["RD_spnd", "Mark_sp", "Admin"], "VIF": [vif_rd, vif_ms, vif_ad]}

Vif_frame= pd.DataFrame(d1)
Vif_frame

# all the variables has low VIF values
# Added varible plot
sm.graphics.plot_partregress_grid(ml_new)

# added varible plot for Mark_spnd is not showing much significance, hence it is excluded
# final model
final_ml= smf.ols("Profit~RD_spnd+Admin", data= strt.new).fit()
final_ml.summary()
final_ml.params

final_pred= final_ml.predict(strt.new)
final_pred

final_pred.corr(strt.new.Profit)

# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)

plt.scatter(strt.new.Profit, final_pred, c="b"); plt.xlabel("Observed values");plt.ylabel("Fitted values")

plt.scatter(final_pred,final_ml.resid_pearson, c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# Looking at the QQ-plot and scatter plot, the final model performs well with high R_square value and significant p-values
from sklearn.model_selection import train_test_split

# Splitting the data into train and test data
strt_train, strt_test= train_test_split(strt.new,test_size=0.2)

strt_train.shape
strt_test.shape

# preparing the model on train data
model_train= smf.ols("Profit~ RD_spnd+Admin", data= strt_train).fit()
model_train.summary()

train_pred = model_train.predict(strt_train)          # train_data prediction

train_resid = train_pred - strt_train.Profit

# RMSE value of train data
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
train_rmse

test_pred= model_train.predict(strt_test)             # test_data prediction

test_resid = test_pred- strt_test.Profit

# RMSE value of test data
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
test_rmse
