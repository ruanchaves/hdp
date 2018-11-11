Here is the output of t-SNE dimensionality reduction applied to 90-dimensional topic vectors produced by a gensim's Hierarchical Dirichlet Process. t-distributed Stochastic Neighbor Embedding is applied twice, once for 90-dimensions to 2D and once for 90-dimensions to 3D. 2D results are interpreted as x,y-coordinates and 3D results are interpreted as colors. Although a human can certainly see the clusters, a computer only knows colored x,y-points so it can't deliver the clusters upon request.

![](https://i.imgur.com/3Zgeqqa.png)

I solved this problem by applying k-nearest neighbors algorithm to the t-SNE result, and after kNN I deleted edges between vertices which had different colors according to a certain degree of tolerance. Then I looked for connected components and I repainted the points according to which connected component they belonged to.

![](https://i.imgur.com/SYz8O0S.png)

Here are the connected components when the tolerance is a little bit lower.

![](https://i.imgur.com/eI8IyhS.png)

Here are the connected components when the tolerance is even lower.

![](https://i.imgur.com/eoIApzU.png)

This means our users won't have to directly deal with this map. When they request a certain document, they'll get a list of similar documents, that is, a list of points sorted according to their distance from the chosen point. And then they'll be able to select cluster colors ( categories ) to filter out the results.

#Related links

[Topic Modeling and t-SNE Visualization](Topic Modeling and t-SNE Visualization)
