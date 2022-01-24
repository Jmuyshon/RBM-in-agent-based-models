from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

#df = pd.read_excel('C:/Users/Leo/Documents/git/savedData/epsVsTr/hn15epo100000centeredTruepeople100interacts100remakeA5.xlsx', index_col=0)

df = pd.read_excel('C:/Users/Leo/Documents/schrijven/res/HN-EPSILON-cd-centered-1000.xlsx', index_col=0)
valueName = 'communicative success'

"""HN-TR-0.5-centered-1000
#sns.lineplot(data=df,x='number of hidden units', y=valueName,hue='number of epochs pretraining',err_style="bars")
sns.barplot(data=df,x='number of hidden units', y=valueName,hue='trainers',capsize=.1)
title = 'different %s for different %s \n versus %s' %('number of hn','trainers',valueName)
plt.title(title)
plt.ylim([0, 1])
plt.show()"""
"""epsvshn
HN-EPSILON-cd-centered-1000"""
#sns.lineplot(data=df,x='number of hidden units', y=valueName,hue='number of epochs pretraining',err_style="bars")
sns.barplot(data=df,x='number of hidden units', y=valueName,hue='epsilon',errcolor='dimgrey',errwidth=1,capsize=.1,palette="PuRd_d")
title = 'different %s for different %s \n versus %s (Learning alg.= CD/epochs=1000/centered=True)' %('number of hidden units','learning rates',valueName)
plt.title(title)
plt.ylim([0, 1])
plt.show()

"""time interval
df = pd.read_excel('C:/Users/Leo/Documents/testen/git/savedData/timesteps/hn15epo100000tr-cdeps0.050000centeredTruepeople100interacts1000remakeA2.xlsx', index_col=0)
valueName = 'communicative success'
sns.lineplot(data=df,x="time interval", y=valueName, err_style="bars")
title = '%s of the intermediate interactions' %(valueName)
plt.title(title)
plt.ylim([0, 1])
plt.show()"""

"""epovshn
#sns.lineplot(data=df,x='number of hidden units', y=valueName,hue='number of epochs pretraining',err_style="bars")
sns.barplot(data=df,x='epsilon', y=valueName,hue='trainers')
title = 'different %s for different %s \n versus %s' %('epsilon','trainers',valueName)
plt.title(title)
plt.ylim([0, 1])
plt.show()"""

"""hnvstr"""
"""sns.barplot(data=df,x='number of hidden units', y=valueName,hue='trainers')
title = 'different %s for different %s \n versus %s' %('number of hidden units','trainers',valueName)
plt.title(title)
plt.ylim([0, 1])
plt.show()"""

"""epovshn"""
"""#sns.lineplot(data=df,x='number of hidden units', y=valueName,hue='number of epochs pretraining',err_style="bars")
sns.barplot(data=df,x='number of hidden units', y=valueName,hue='number of epochs pretraining')
title = 'different %s for different %s \n versus %s' %('number of hidden units','number of epochs pretraining',valueName)
plt.title(title)
plt.ylim([0, 1])
plt.show()"""

"""IL epovshn"""
"""valueName = 'communicative success'
sns.catplot(data=df,x='number of hidden units', y=valueName,hue='number of epochs pretraining',col="agent which evaluates",kind="bar")
#sns.barplot(data=result,x=param1, y=valueName,hue=param2)
title = 'different hn for different epochs \n versus %s' %(valueName)
#plt.title(title)
plt.ylim([0, 1])
plt.show()
#1: hidden units"""
""" 
sns.lineplot(data=df,x="number of hidden units", y="success",err_style="bars")
plt.title('number of hidden units versus success')
plt.ylim([0, 1])
plt.show()"""

#2: trainers
""" 
sns.barplot(data=df,x="trainers", y="success")
plt.title('trainers versus success')
plt.ylim([0, 1])
plt.show()"""

#3.1: epsilon trainer
""" 
sns.lineplot(data=df,x="epsilon", y="success",err_style="bars")
title = "epsilon for %s versus success" %(tr)
plt.title(title)
plt.ylim([0, 1])
plt.show()"""

#3.2: epsilon different trainers
""" 
sns.lineplot(data=result,x="epsilon", y="success",hue="trainer")
plt.title('testing the optimal epsilon for the different kind of trainers')
plt.show()"""

#4: centerd?
""" 
sns.barplot(data=df,x="centered?", y="success")
plt.title('comparing centered and normal RBM')
plt.ylim([0, 1])
plt.show()"""

#5: epochs
""" 
sns.lineplot(data=df,x="number of epochs pretraining", y="success",err_style="bars")
plt.title('number of epochs pretraining versus success')
plt.ylim([0, 1])
plt.show()"""