# Classification Using Principal Components Analysis (PCA)
## Identify classes by proximity to first eigen vectors.

Principal Components Analysis (PCA) is a statistical technique commonly used in data science to reduce dimensionality in feature sets. There are excellent Towards Data Science articles dedicated to the conceptual explanation and use of PCA in feature selection (1) (2). The goal of this article is to develop an algorithm where PCA is used in Machine Learning classification problems.

A new data point in feature space can be classified by measuring its proximity to the centroids of class groups. 

### Classification based on proximity to centroids

![Classification based on proximity to centroids](/assets/images/Classification-Centroid.png)

To implement classification based on proximity in Python, we first need to compute group-related statistics:

```python
# first, compute statistics about class groups
groups = dict()
group_list= ["green", "orange"]
group_col= "label"
groupby = df.groupby(label)
for g in groups:
  groups[g] = dict()
  groups[g]['data'] = groupby.get_group(g).drop(group_col, axis=1)
  groups[g]['df'] = pd.DataFrame(np.array(groups[g]['data']))
  groups[g]['pca_fit'] = PCA(n_components=n_components).fit(groups[g]['df'])
  groups[g]['std'] = np.sqrt(groups[g]['pca_fit'].explained_variance_)
  groups[g]['mean'] = groups[g]['pca_fit'].mean_
```

Classification based on proximity to centroids could be implemented by:

```python
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
```

If the class groups exhibit elongated patterns, the variances and directions in these patterns could be robustly captured by the PCA eigen values and vectors. The proximity to the component axes defined by the first eigen vectors could be used for classification purposes.

### Classification based on proximity to first principal components

![Classification based on proximity to first principal components](/assets/images/Classification-PCA.png)

Classification based on proximity to first principal components could be implemented by:

```python
distances = dict()
group_df = pd.DataFrame(NewData)

# use pca_fit computed previously to transform the NewPoint 
# from the feature space into the principal component space
for g in groups:
  distances[g] = pd.DataFrame(data= groups[g]['pca_fit'].transform(princ_df), 
                  columns=['col'+str(i) for i in range(n_components)])
dist_min = np.inf
result = ''
for d in distances:
  dis = abs(distances[d]['PC'+str(1)].loc[0]) # PC1
  if dis < dist_min:
    dist_min = dis
    result = d
```

Since an elongated group shape is a prerequisite for the applicability of the current PCA-based classification algorithm, a criterion is needed for determining if a group cluster is sufficiently elongated. The subject of "unidimensionality" is investigated in the field of Psychology and a ratio of greater than 3 to 5 between the first and second eigen values is generally accepted as existence of elongation (3).

```python
# all ratios in ratio_maxs must be larger than a threshold value, like 3.

ratio_maxs = [ groups[g]['std'][0]/groups[g]['std'][1] for g in classes ]
```

The last consideration in the development of the algorithm is the selection of the feature set which exhibits the largest elongation, that is the maximum ratio between the first and second eigen values. An alternative approach where the number of features and data points are very large would be run the algorithm against an exhaustive list of feature sets and select the feature set that return the highest accuracy.

---

A computationally performant method is introduced in this article that effectively identifies the classes of new data points. The negative aspect of this novel ML algorithm is that it can be used only with data sets that exhibit elongated class group structures. The feature sets that can provide the necessary elongation can be identified by investigation of the ratios between the first and second eigen values.

---

References:
1. [Principal Component Analysis (PCA) Explained as Simple as Possible](https://medium.com/@mattisonhineline/principal-component-analysis-pca-explained-as-simple-as-possible-896fd8dee9c)
2. [Dimensionality Reduction (PCA) Explained](https://towardsdatascience.com/dimensionality-reduction-explained-5ae45ae3058e)
3. [A Note on Using Eigenvalues in Dimensionality Assessment](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1358&context=pare#:~:text=Parallel%20analysis%20determines%20the%20number,thus%20define%20a%20valid%20dimension.)
