# Importing the necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
corolla = pd.read_csv("ToyotaCorolla.csv", encoding= 'unicode_escape')
corolla.head(1)
corolla.shape
corolla.columns

# considering only the important variables
cor = corolla.loc[ : ,["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
cor.head()

# Renaming the columns for convinience
cor= cor.rename(columns={"Age_08_04":"Age","Quarterly_Tax": "Quar_tax"})
cor.columns             # column names

# Correlation matrix
cor.corr()

 import seaborn as sns
 # Scatter plot between the variables along with histograms
sns.pairplot(cor)

import statsmodels.formula.api as smf
# preparing model considering all the variables
ml1= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quar_tax+Weight", data= cor).fit()
ml1.params                  # Getting coefficients of variables
ml1.summary()

# preparing model based only on cc
ml_cc= smf.ols("Price~cc", data= cor).fit()
ml_cc.params
ml_cc.summary()

# preparing model based only on Doors
ml_dr= smf.ols("Price~Doors",data= cor).fit()
ml_dr.params
ml_dr.summary()

# preparing model based on Market_spnd and Admin
ml_cd= smf.ols("Price~Doors+cc", data=cor).fit()
ml_cd.params
ml_cd.summary()

# Both coefficients p-value are significant when used togther without other variables
import statsmodels.api as sm
# Checking whether data has any influential values
sm.graphics.influence_plot(ml1)              # influence index plots

# index 80 AND 221 is showing high influence so we can exclude that entire row
cor_new=cor.drop(cor.index[[80,221]],axis=0)

# Preparing new model
ml_new= smf.ols("Price~Age+KM+HP+Quar_tax+Weight+cc+Gears+Doors", data=cor_new).fit()
ml_new.params            # Getting coefficients of variables
ml_new.summary()

print(ml_new.conf_int(0.05))          # for 95% confidence interval

# predicted values of Price
pred = ml_new.predict(cor_new)
pred

# calculating VIF's values of independent variables
rsq_ag =smf.ols("Age~KM+HP+cc+Doors+Gears+Quar_tax+Weight",data=cor_new).fit().rsquared
vif_ag= 1/(1-rsq_ag)
vif_ag

rsq_km =smf.ols("KM~Age+HP+cc+Doors+Gears+Quar_tax+Weight",data=cor_new).fit().rsquared
vif_km= 1/(1-rsq_km)
vif_km

rsq_hp =smf.ols("HP~Age+KM+cc+Doors+Gears+Quar_tax+Weight",data=cor_new).fit().rsquared
vif_hp= 1/(1-rsq_hp)
vif_hp

rsq_cc =smf.ols("cc~Age+KM+HP+Doors+Gears+Quar_tax+Weight",data=cor_new).fit().rsquared
vif_cc= 1/(1-rsq_cc)
vif_cc

rsq_dr =smf.ols("Doors~Age+KM+HP+cc+Gears+Quar_tax+Weight",data=cor_new).fit().rsquared
vif_dr= 1/(1-rsq_dr)
vif_dr

rsq_gr =smf.ols("Gears~Age+KM+HP+cc+Doors+Quar_tax+Weight",data=cor_new).fit().rsquared
vif_gr= 1/(1-rsq_gr)
vif_gr

rsq_qt =smf.ols("Quar_tax~Age+KM+HP+cc+Doors+Gears+Weight",data=cor_new).fit().rsquared
vif_qt= 1/(1-rsq_qt)
vif_qt

rsq_wt =smf.ols("Weight~Age+KM+HP+cc+Doors+Gears+Quar_tax",data=cor_new).fit().rsquared
vif_wt= 1/(1-rsq_wt)
vif_wt

# Storing vif values in a data frame
d1 = {'Variables':['KM','HP','CC','Doors','Gears','Quar_tax','Weight'], 'VIF':[vif_km,vif_hp,vif_cc,vif_dr,vif_gr,vif_qt,vif_wt]}
d1
Vif_frame = pd.DataFrame(d1)
Vif_frame


# all the variables has low VIF values
# Added varible plot
sm.graphics.plot_partregress_grid(ml_new)

# added varible plot for Doors is not showing much significance, hence it is excluded
# final model
final_ml = smf.ols("Price~Age+KM+HP+cc+Gears+Quar_tax+Weight",data=cor_new).fit()
final_ml.summary()               # Getting coefficients of variables

# predicted values of price using final model
pred_final= final_ml.predict(cor_new)
pred_final

# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)

plt.scatter(cor_new.Price, pred_final, c="b");plt.xlabel("Observed values");plt.ylabel("Fitted values")

pred_final.corr(cor_new.Price)              # Correlation between the predicted and actual values

plt.scatter(pred_final,final_ml.resid_pearson, c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
# PLot showing Residuals VS Fitted Values

from sklearn.model_selection import train_test_split
# Splitting the data into train and test data

cor_train, cor_test= train_test_split(cor_new, test_size=0.2)

# preparing the model on train data
model_train=smf.ols("Price~Age+KM+HP+cc+Gears+Quar_tax+Weight", data=cor_train).fit()
model_train.summary()

train_pred= model_train.predict(cor_train)           # train_data prediction
train_resid = train_pred - cor_train.Price

# RMSE value of train data
train_rmse= np.sqrt(np.mean(train_resid*train_resid))
train_rmse

test_pred= model_train.predict(cor_test)             # test_data prediction
test_resid= test_pred- cor_test.Price

# RMSE value of test data
test_rmse= np.sqrt(np.mean(test_resid*test_resid))
test_rmse
