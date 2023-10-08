
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def classifier_by_centroid(new_data):
    # calculates distances to centroids
    distances = [np.linalg.norm(new_data - groups[g]['mean']) for g in classes]

    # select the class corresponding to minimum distance to centroid
    dist_min = np.inf
    result = ''
    for g, d in zip(classes, distances):
        if d < dist_min:
            dist_min = d   
            result = g
    return result

def classifier_by_pca(data):
  distances = dict()
  princ_df = pd.DataFrame(data, columns=features)

  # calculates distances to group principal components
  for g in groups:
    distances[g] = pd.DataFrame(data= groups[g]['pca_fit'].transform(princ_df), 
                                columns=['col'+str(i) for i in range(n_components)])

  # select the class with minimum distance to eigen vector
  # if the projection on the eigen vector < 2*std, it's a strong prediction.
  # otherwise, it is a weak prediction.
  dis_weak = np.inf
  dis_strong = np.inf
  res_weak = ''
  res_strong = ''
  for d in distances:
    dis = abs(distances[d]['col'+str(1)].loc[0])
    loc = abs(distances[d]['col'+str(0)].loc[0])
    std = groups[d]['std'][0]
    if loc < std*2:
      if dis < dis_strong:
        dis_strong = dis
        res_strong = d
    else:
      if dis < dis_weak:
        dis_weak = dis     
        res_weak = d

  return  res_strong if res_strong!='' else res_weak

###################################################

df = pd.DataFrame([ [6.1,3.0,'G'],
                    [6.05,2.1,'G'],
                    [6,1.2,'G'],

                    [5,3.5,'O'],
                    [6.1,4.2,'O'],
                    [7,5.0,'O']], columns=['f1','f2','label'])

n_components = len(df.columns)-1 # no. of features (also components)
label = df.columns[len(df.columns)-1] # 'label'
features = list(df.columns[0:len(df.columns)-1]) # ['f1', 'f2']
classes = list(df.groupby(label).groups.keys()) # ['G', 'O']

scaler = StandardScaler().fit(df[features])
df[features] = scaler.transform(df[features])

groups = dict()
for g in classes:
  groups[g] = dict()
  groups[g]['data'] = df.groupby(label).get_group(g).drop(label, axis=1)
  groups[g]['df'] = pd.DataFrame(np.array(groups[g]['data']), columns=features)
  groups[g]['pca_fit'] = PCA(n_components=n_components).fit(groups[g]['df'])
  groups[g]['std'] = np.sqrt(groups[g]['pca_fit'].explained_variance_)
  groups[g]['mean'] = groups[g]['pca_fit'].mean_

data = pd.DataFrame([[6.2, 3.8]], columns=features)
data_std = scaler.transform(data)

print(classifier_by_centroid(data_std))
print(classifier_by_pca(data_std))

# all ratios in ratio_maxs must be larger than a threshold value, like 3.
ratio_maxs = [ groups[g]['std'][0]/groups[g]['std'][1] for g in classes ]
print(ratio_maxs)
# %%
