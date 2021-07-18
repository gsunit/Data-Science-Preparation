# Data Science Preparation

**P.S. Ctrl+F to serach for relevant keywords.**

### Preliminaries

If you are just beginning with ML & Data Science, a good first place to start will be 
 - [ ] [Andrew Ng Coursera ML course](https://www.coursera.org/learn/machine-learning). Finish at least the first few weeks.

If you have already done the Andrew Ng course, you might want to brush up on the concepts through these notes.
 - [ ] [Notes on Andrew Ng Machine Learning](https://www.holehouse.org/mlclass/)

If you want to make a list of important interview topics head over to this article.
 - [ ] [Machine Learning Cheatsheet](https://medium.com/swlh/cheat-sheets-for-machine-learning-interview-topics-51c2bc2bab4f)

### Courses & Resources
 - [ ] [Become a Data Scientist in 2020 with these 10 resources](https://towardsdatascience.com/top-10-resources-to-become-a-data-scientist-in-2020-99a315194701)
 - [ ] [Applied Data Science with Python | Coursera](https://www.coursera.org/specializations/data-science-python?ranMID=40328&ranEAID=lVarvwc5BD0&ranSiteID=lVarvwc5BD0-_4L3mvw.I6oY9SNPHAtR2Q&siteID=lVarvwc5BD0-_4L3mvw.I6oY9SNPHAtR2Q&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0)
 - [ ] [Minimal Pandas Subset for Data Scientists - Towards Data Science](https://towardsdatascience.com/minimal-pandas-subset-for-data-scientists-6355059629ae)
 - [ ] [Python’s One Liner graph creation library with animations Hans Rosling Style](https://towardsdatascience.com/pythons-one-liner-graph-creation-library-with-animations-hans-rosling-style-f2cb50490396)
 - [ ] [3 Awesome Visualization Techniques for every dataset](https://towardsdatascience.com/3-awesome-visualization-techniques-for-every-dataset-9737eecacbe8)
 - [ ] [Inferential Statistics | Coursera](https://www.coursera.org/learn/inferential-statistics-intro?ranMID=40328&ranEAID=lVarvwc5BD0&ranSiteID=lVarvwc5BD0-ydEVG6k5kidzLtNqbbVQvQ&siteID=lVarvwc5BD0-ydEVG6k5kidzLtNqbbVQvQ&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0)
 - [ ] [Advanced Machine Learning | Coursera](https://www.coursera.org/specializations/aml?ranMID=40328&ranEAID=lVarvwc5BD0&ranSiteID=lVarvwc5BD0-_1LkRNzPhJ43gzMHQzcbag&siteID=lVarvwc5BD0-_1LkRNzPhJ43gzMHQzcbag&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0)
 - [ ] [Deep Learning | Coursera](https://www.coursera.org/specializations/deep-learning?ranMID=40328&ranEAID=lVarvwc5BD0&ranSiteID=lVarvwc5BD0-m3SBadPJeg1Z1rWVng39OQ&siteID=lVarvwc5BD0-m3SBadPJeg1Z1rWVng39OQ&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0)
 - [ ] [Deep Neural Networks with PyTorch | Coursera](https://www.coursera.org/learn/deep-neural-networks-with-pytorch?ranMID=40328&ranEAID=lVarvwc5BD0&ranSiteID=lVarvwc5BD0-Kb0qPiTtTFPC3kMQZlnqpg&siteID=lVarvwc5BD0-Kb0qPiTtTFPC3kMQZlnqpg&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0)
 - [ ] [Machine Learning - complete course notes](http://www.holehouse.org/mlclass/)
 - [ ] [Data Science Interview Questions | Data Science Interview Questions and Answers with Tips](https://www.youtube.com/watch?v=7YuTmLvs1Dc)

### Data Science Practice Questions
If you are clueless about which topic to start from in data science, but have some basic idea about ML, then simply give these questions a go. If you get a bunch of them wrong, you'll know where to start your preparation :)

### SQL
Quickly go through the tutorial pages, you need not cram anything. Soon after, solve all the Hackerrank questions (in sequence, without skipping). Refer back to any of the tutorials or look up the discussion forum when stuck. You will learn more effectively this way and applying the various clauses will boost your recall.

- [ ] [SQL Tutorial Series](https://www.w3schools.com/sql/default.asp)
- [ ] [Hackerrank SQL Practice Questions](https://www.hackerrank.com/domains/sql)
- [ ] [Interview Questions - SQL Nomenclature, Theory, Databases](https://www.jigsawacademy.com/blogs/business-analytics/sql-joins-interview-questions/)
- [ ] [SQL Joins](https://learnsql.com/blog/sql-join-interview-questions-with-answers/)
- [ ] [Popular Interview Questions solved](https://github.com/Aafreen29/SQL-Interview-Prep-Question/blob/master/queries.sql)
- [ ] [Amazon Data Analyst SQL Interview Questions](https://leetcode.com/discuss/interview-question/606844/amazon-data-analyst-sql-interview-questions)

### Probability
//TODO: add probability resources here. Accepting PRs.

### Statistics

<details>
  <summary>Why divide by n-1 in sample standard deviation</summary>
 
  - Let f(v) = sum( (x_i-v)^2 )/n. Using f'(v) = 0, minima occurs at v = sum(x_i)/n = sample mean
  - Thus, f(sample mean) < f(population mean), as minima occurs at sample mean
  - Thus, sample std < population std (when using n in denominator)
  - But our goal was to estimate a value close to population std using the data of samples.
  - So we bump us sample std a bit by decreasing its denominator to n-1. Thus, bringing sample std closer to population std
                                       
 
</details>

 - [ ] [Variance, Standard Deviation, Covariance, Correlation](https://www.mygreatlearning.com/blog/covariance-vs-correlation/)
 - [ ] [Probability vs Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4)
 - [ ] [Maximum Likelihood For the Normal Distribution, step-by-step!](https://www.youtube.com/watch?v=Dn6b9fCIUpM)
 - [ ] [Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
 - [ ] [Why Dividing By N Underestimates the Variance](https://www.youtube.com/watch?v=sHRBg6BhKjI)
 - [ ] [The Central Limit Theorem](https://www.youtube.com/watch?v=YAlJCEDH2uY)
 - [ ] [Gaussian Naive Bayes](https://www.youtube.com/watch?v=H3EjCKtlVog)
 - [ ] [Covariance and Correlation Part 1: Covariance](https://www.youtube.com/watch?v=qtaqvPAeEJY)
 - [ ] [Expectation Maximization: how it works](https://www.youtube.com/watch?v=iQoXFmbXRJA)
 - [ ] [Bayesian Inference: An Easy Example](https://www.youtube.com/watch?v=I4dkEALQv34)

### Linear Algebra
 - [ ] [Eigenvectors and eigenvalues | Essence of linear algebra, chapter 14](https://m.youtube.com/watch?feature=youtu.be&v=PFDu9oVAE-g)

### Distributions
 - [ ] [(1) Exponential and Laplace Distributions](https://www.youtube.com/watch?v=5ptp4naoYEo)
 - Gamma
 - Exponential
 - Students' T
 
### Inferential Statistics

<details>
  <summary>Notes on p-values, statistical significance</summary>
 
 - p-values
 
    - 0 <= p-value <= 1
   
    - The closer the p-value to 0, the more the confidence that the null hypothesis (that there is no difference between two things) is false.
   
    - `Threshold for making the decision`: 0.05. This means that if there is no difference between the two things, then and the same experiment is repeated a bunch of times, then only 5% of them would yield a wrong decision.
   
    - In essence, 5% of the experiments, where the differences come from weird random things, will generate a p-value less that 0.05.
   
    - Thus, we should obtain large p-values if the two things being compared are identical.
   
    - Getting a small p-value even when there is no difference is known as a False positive.'
   
    - If it is extremely important when we say that the two things are different, we use a smaller threshold like 0.1%.
   
    - A small p-value does not imply that the difference between the two things is large.
    
 - Error Types
    
    - `Type-1 error`: Incorrectly reject null (False positive)
   
    - `Alpha`: Prob(type-1 error) (aka level of significance)
   
    - `Type-2 error`: Fail to reject when you should have rejected null hypothesis (False negative)
   
    - `Beta`: Prob(type-2 error)
   
    - `Power`: Prob(Finding difference between when when it truly exists) = 1 - beta
   
    - Having power > 80% for a study is good. Calculated before study is conducted based on projections.
   
    - `P-value`: Prob(obtaining a result as extreme as the current one, assuming null is true)
   
    - Low p-value -> reject null hypothesis, high p-value -> fail to reject hypothesis
   
    - If p-value < alpha -> study was statistically significant. Alpha = 0.05 usually
    
</details>

 
 - [ ] [Null Hypothesis, p-Value, Statistical Significance, Type 1 Error and Type 2 Error](https://www.youtube.com/watch?v=YSwmpAmLV2s)
 - [ ] [Hypothesis Testing and The Null Hypothesis](https://www.youtube.com/watch?v=0oc49DyA3hU)
 - [ ] [How to calculate p-values](https://www.youtube.com/watch?v=JQc3yx0-Q9E)
 - [ ] [P Values, clearly explained](https://www.youtube.com/watch?v=5Z9OIYA8He8)
 - [ ] [p-values: What they are and how to interpret them](https://www.youtube.com/watch?v=vemZtEM63GY)
 - [ ] [Intro to Hypothesis Testing in Statistics - Hypothesis Testing Statistics Problems &amp; Examples](https://www.youtube.com/watch?v=VK-rnA3-41c)
 - [ ] [Idea behind hypothesis testing | Probability and Statistics](https://www.youtube.com/watch?v=dpGmVV0-4jc)
 - [ ] [Examples of null and alternative hypotheses | AP Statistics](https://www.youtube.com/watch?v=_3_6wjycJdk)
 - [ ] [Confidence Intervals](https://www.youtube.com/watch?v=TqOeMYtOc1w)
 - [ ] [P-values and significance tests | AP Statistics](https://www.youtube.com/watch?v=KS6KEWaoOOE)
 - [ ] [Feature selection — Correlation and P-value | by Vishal R | Towards Data Science](https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf)
 
### Statistical Tests

 <details>
  <summary>t-Test</summary>

    - compares 2 means. Works well when sample size is small. We esimate popl_std by sample_std.

    - We are less confident that the distribution resembles normal dist. As sample size increases, it approches normal dist (at about n~=30)

    - t-value = signal/noise = (absolute diff bet two means)/(variability of groups) = | x1 - x2 | / sqrt(s1^2/n1  +  s2^2/n2)

    - Thus, increasing variance will give you more noise. Increasing #samples will decrease the noise.

    - Degrees of freedom (DOF) = n1 + n2 - 2

    - if t-value > critical value (from table) => reject hypothesis (found a statistically significant diff bet two means) 

    - Independent (unpaired) samples means that two separate populations used to take samples. Paired samples means samples taken from the same population, and now we are comparing two means.

    - In a two tailed test, we are not sure which direction the variance will be. Considering alpha=0.05, the 0.05 is split into 0.025 on both of the tails. In the middle is the remaining 0.95. Run a one-tailed test if sure about the directionality.

    - Calculating t-statistic when comparing sample mean with an already known mean. t-statistic = 

</details>

<details>
  <summary>Z-test</summary>
    
    - Uses a normal distribution

    - (mu, sigma) are population statistics. (x_bar, s) are sample statistics. 

    - z-score = (x-mu)/sigma
    - z-statistic = (x_bar - mu)/ sqrt(sigma^2/n)
    - t-statistic = (x_bar - mu)/ sqrt(s^2/n)

    - Use z-stat when pop_std (sigma) is known and n>=30. Otherwise use t-stat.
    
</details>

<details>
  <summary>Chi-squred test</summary>

    - chi^2 = sum( (observed-expected)^2 / (expected) )
    - The larger the chi^2 value, the more likely the variables are related
    - Correlation relationship between two attributes, A and B. A has c distinct values and B has r
    - Contingency table: c values of A are the columns and r values of B the rows
    - (Ai ,Bj): joint event that attribute A takes on value ai and attribute B takes on value bj
    - oij= observed frequency, eij= expected frequency
    - Test is based on a significance level, with (r -1)x(c-1) degrees of freedom
    - Slides link: https://imgur.com/a/U4uJhHc

</details>

<details>
  <summary>Statistical Tests notes</summary>

   - ANOVA test: compares >2 means
   - Chi-squared test: compares categorical variables
   - Shapiro Wilk test: test if a random sample comes from a normal distribution
   - Kolmogorov-Smirnov Goodness of Fit test: compares data with a known distribution to check if they have the same distribution

</details>
 
 - [ ] [Student's t-test](https://www.youtube.com/watch?v=pTmLQvMM-1M)
 - [ ] [Z-Statistics vs. T-Statistics](https://www.youtube.com/watch?v=DEkPZv5ppHI)
 - [ ] [Hypothesis Testing Problems Z Test & T Statistics One & Two Tailed Tests 2](https://www.youtube.com/watch?v=zJ8e_wAWUzE)
 - [ ] [Contingency table chi-square test | Probability and Statistics](https://www.youtube.com/watch?v=hpWdDmgsIRE)
 - [ ] [6 ways to test for a Normal Distribution — which one to use? (Kolmogorov Smirnov test, Shapiro Wilk test)](https://towardsdatascience.com/6-ways-to-test-for-a-normal-distribution-which-one-to-use-9dcf47d8fa93)

### Linear Regression & Logistic Regression
 - [ ] [Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
 - [ ] [R-squared or coefficient of determination | Regression | Probability and Statistics](https://www.youtube.com/watch?v=lng4ZgConCM)
 - [ ] [Linear Regression vs Logistic Regression | Data Science Training | Edureka](https://www.youtube.com/watch?v=OCwZyYH14uw)
 - [ ] [Regression and R-Squared (2.2)](https://www.youtube.com/watch?v=Q-TtIPF0fCU)
 - [ ] [Linear Models Pt.1 - Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
 - [ ] [How To... Perform Simple Linear Regression by Hand](https://www.youtube.com/watch?v=GhrxgbQnEEU)
 - [ ] [Missing Data Imputation using Regression | Kaggle](https://www.kaggle.com/shashankasubrahmanya/missing-data-imputation-using-regression)
 - [ ] [Covariance and Correlation Part 2: Pearson&#39;s Correlation](https://www.youtube.com/watch?v=xZ_z8KWkhXE)
 - [ ] [R-squared explained](https://www.youtube.com/watch?v=2AQKmw14mHM)
 
### Precision, Recall
<details>
  <summary>Important Formulae</summary>
  
   - `Sensitivity`   = True Positive Rate = TP/(TP+FN)                         = how sensitive is the model, same as recall
   - `Specificity`   = 1 - False Positive Rate = 1 - FP/(FP+TN) = TN/(FP+TN) 
   - `'P'recision`   = TP/(TP+FP) = TP / 'P'redicted Positive                  = how less often does the model raise a false alarm
   - `'R'ecall`      = TP/(TP+FN) = TP / 'R'eal Positive                       = of all the true cases, how many did we catch
   - `F1-score`      = 2*Precision*Recall/(Precision + Recall)                 = geometric mean of precision & recall

</details>

 - [ ] [ROC and AUC!](https://www.youtube.com/watch?v=4jRBRDbJemM)
 - [ ] [How to Use ROC Curves and Precision-Recall Curves for Classification in Python](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)
 - F1 score, specificity, sensitivity

### Gradient Descent
 - [ ] [Stochastic Gradient Descent](https://www.youtube.com/watch?v=vMh0zPT0tLI)
 
### Decision Trees & Random Forests
 
 <details>
   <summary>Information Gain</summary>
  
   - Information gain determines the reduction of the uncertainty after splitting the dataset on a particular feature such that if the value of information gain increases, that feature is most useful for classification.
   - IG = entropy after splitting - entropy before spliting
   - Entropy =  - sum_over_n ( p_i * ln2(p_i) )
 </details>
 
 <details>
   <summary>Gini Index</summary>
  
   - Higher the GI, more randomness. An attribute/feature with least gini index is preferred as root node while making a decision tree. 
   - 0: all elements correctly divided
   - 1: all elements randomly distributed across various classes
   - 0.5: all elements uniformly distributed into some classes
   - GI (P) = 1 - sum_over_n(p_i^2) where
   - P=(p1 , p2 ,.......pn ) , and pi is the probability of an object that is being classified to a particular class.
  
 </details>
 
 - [ ] [Decision and Classification Trees](https://www.youtube.com/watch?v=_L39rN6gz7Y)
 - [ ] [Regression Trees](https://www.youtube.com/watch?v=g9c66TUylZ4)
 - [ ] [Gini Index, Infromation Gain](https://www.analyticssteps.com/blogs/what-gini-index-and-information-gain-decision-trees)
 - [ ] [Decision Trees, Part 2 - Feature Selection and Missing Data](https://www.youtube.com/watch?v=wpNl-JwwplA)
 - [ ] [How to Prune Regression Trees](https://www.youtube.com/watch?v=D0efHEJsfHo)
 - [ ] [Random Forests Part 1 - Building, Using and Evaluating](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
 - [ ] [Python | Decision Tree Regression using sklearn - GeeksforGeeks](https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/?ref=rp)
 
### Loss functions
 - [ ] [Why do we need Cross Entropy Loss? (Visualized)](https://www.youtube.com/watch?v=gIx974WtVb4)
 - [ ] [Cross-entropy loss (Binary, Multi-Class, Multi-Label)](https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451)

### L1, L2 Regression
 - [ ] [Ridge vs Lasso Regression, Visualized](https://www.youtube.com/watch?v=Xm2C_gTAl8c)
 - [ ] [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30)
 - [ ] [Regularization Part 2: Lasso (L1) Regression](https://www.youtube.com/watch?v=NGf0voTMlcs)
 - [ ] [Regularization Part 3: Elastic Net Regression](https://www.youtube.com/watch?v=1dKRdX9bfIo)
 - [ ] [regression - Why is the L2 regularization equivalent to Gaussian prior? - Cross Validated](https://stats.stackexchange.com/questions/163388/why-is-the-l2-regularization-equivalent-to-gaussian-prior)
 - [ ] [regression - Why L1 norm for sparse models - Cross Validated](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models)
 
### PCA, SVM, LDA
 
 <details>
  <summary>PCA</summary>
    
    - Create a covariance matrix of the variables. Its eigenval and eigenvec describe the full multi-dimensional dataset.
    - Eigenvec describe the direction of spread, Eigenval describe the importance of certain directions in describing the spread.
    - In PCA, sequentially determine the axes in which the data varies the most
    - All selected axes are eigenvectors of the symmetric covariance matrix, thus they are mutually perpendicular
    - Then reframe the data using a subset of the most influential axes, by plotting the projections of original points on these axes. Thus dimensional reduction.
    - Singular Value Decomposition is a way to find those vectors 
    
</details>
 
<details>
  <summary>SVM</summary>
  
    - Margin is the smallest distance between decision boundary and data point.
    - Maximum margin classifiers classify by using a decision boundary placed such that margin is maximized. Thus, they are super sensitive to outliers.
    - Thus, when we allow some misclassifications to accomodate outliers, it is know as a Soft Margin Classifier aka Support Vector Classifier (SVC).
    - Soft margin is determined through cross-validation. Support Vectors are those observations on the edge of Soft Margin.
    - For 3D data, the Support Vector Classifier forms a plane. For 2D it forms a line.
    - Support Vector Machines (SVM) moves the data into a higher dimension (new dimensions added by applying transformation on original dimensions)
    - Then, a support vector classifier is found that separates the higher dimensional data into two groups.
    - SVMs use Kernels that systematically find the SVCs in higher dimensions.
    - Say 2D data transformed to 3D. Then Polynomial Kernels find 3D relationships between each pair of those 3D points. Then use them to find an SVC.
    - Radial Basis Function (RBF) Kernel finds SVC in infinite dimensions. It behavs like a weighted nearest neighbour model (closest observations have the most impact on classification)
    - Kernel functions do not need to transform points to higher dimenstion. They find pair-wise relationship between points as if they were in higher dimensions, known as Kernel Trick
 
    
</details>
 
 - [ ] [PCA main ideas in only 5 minutes](https://www.youtube.com/watch?v=HMOI_lkzW08)
 - [ ] [Visual Explanation of Principal Component Analysis, Covariance, SVD](https://www.youtube.com/watch?v=5HNr_j6LmPc)
 - [ ] [Principal Component Analysis (PCA), Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ)
 - [ ] [Support Vector Machines](https://www.youtube.com/watch?v=efR1C6CvhmE)
 - [ ] [Linear Discriminant Analysis (LDA) clearly explained.](https://www.youtube.com/watch?v=azXCzI57Yfc)
 
### Boosting
 
 <details>
  <summary>Notes on Adaboost, Gradient Boosting, XGBoost</summary>
  
   - Adaboost
    
      - Combines a lot of "weak learners" to make decisions.
    
      - Single level decision trees (one root, two leaves), known as stumps.
    
      - Each stump has a weighted say in voting (as opposed to random forests where each tree has an equal vote).
    
      - Errors that the first stump makes, influences how the second stump is made. Thus, order is important
    (as opposed to random forests where each tree is made independent of the others, doesnt matter the order in which trees are made)
    
      - First all samples are given a weight (equal weights initially). Then first stump is made based on which feature classifies the best
    (feature with lowest Gini index chosen). Now to decide stump's weight in final classification, we calculate the following. 
    
      - total_error = sum(weights of samples incorrectly classified)
      - amount_of_say = 0.5log( (1-total_error)/total_error )
    
      - When stump does a good job, amount_of_say is closer to 1.
    
      - Now modify the weights so that the next stump learns from the mistakes.
      - We want to emphasize on correctly classifying the samples that were wronged earlier.
    
      - new_sample_weight = sample_weight * exp(amount_of_say) => increased sample weight
      - new_sample_weight = sample_weight * exp(-amount_of_say) => decreased sample weight
    
      - Then normalize new_sample_weights. Then create a new collection by sampling records, but with a greater probablilty of
    picking those which were wrongly classified earlier. This is where you can use new_sample_weights (normalized).
    After re-sampling is done, assign equal weights to all samples and repeat for finding second stump. 
    
   - Gradient Boost
    
      - Starts by making a single leaf instead of a stump. Considering regression, leaf contains average of target variable as initial prediction.
    
      - Then build a tree (usu with 8 to 32 leaves). All trees are scaled equally (unlike AdaBoost where trees are weighted while prediciton)
    
      - The successive trees are also based on previous errors like AdaBoost.
    
      - Using initial prediction, calculate distance from actual target values, call them residuals, and store them.
    
      - Now use the features to predict the residuals. The average of the values that finally end up in the same leaf is used as the
    predicted regression value for that leaf (this is true when the underlying loss function to be minimized is the squared residual fn.)
    
      - Then 
      - new_prediction = initial_prediction + learning_rate*result_from_tree1
      - new_residual = target_value - new_prediction
    
      - new_residual will be smaller than old_residual, thus we are taking small steps towards learning to predict target_value accurately
    
      - Train new tree on the new_residual, add the result_from_tree2*learning_rate to new_prediction to update it. Rinse and repeat.

  
</details>

 - [ ] [Gradient Boost, Learning Rate Shrinkage](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
 - [ ] [Gradient Boost Part 1: Regression Main Ideas](https://www.youtube.com/watch?v=3CC4N4z3GJc)
 - [ ] [XGBoost Part 1: Regression](https://www.youtube.com/watch?v=OtD8wVaFm6E)
 - [ ] [AdaBoost](https://www.youtube.com/watch?v=LsK-xG1cLYA)
 

### Quantiles
 - [ ] [Quantile-Quantile Plots (QQ plots)](https://www.youtube.com/watch?v=okjYjClSjOg)
 - [ ] [Quantiles and Percentiles](https://www.youtube.com/watch?v=IFKQLDmRK0Y)

### Clustering
 - [ ] [Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo)
 - [ ] [K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)

### Neural Networks
 
 <details>
  <summary>CNN notes</summary>
  
  - for data with grid like topology (1D audio, 2D image)
  - reduces params in NN through
    - sparse interactions
    - parameter sharing 
      - CNN creates spatial features. 
      - Image passed through CNN gives rise to a volume. Section of this volume taken through the depth represents features of the same part of image
      - Each feature in the same depth layer is generated by the same filter that convolves the image (same kernel, shared parameters)
    - equivariant representation
      - f(g(x)) = g(f(x))
  - Types of layers
    - Convolution layer - image convolved using kernels. Kernel applied through a sliding window. Depth of kernel = 3 for RGB image, 1 for grey-scale
    - Activation Layer
  
 </details>
 
 - [ ] [But what is a neural network? | Chapter 1, Deep learning] (https://www.youtube.com/watch?v=aircAruvnKk)
 - [ ] [Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w)
 - [ ] [What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U) 
 - [ ] [Train-test splitting, Stratification](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)
 - [ ] [Regularization, Dropout, Early Stopping](https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/)
 - [ ] [Convolution Neural Networks - EXPLAINED](https://www.youtube.com/watch?v=m8pOnJxOcqY)
 - [ ] [k-fold Cross-Validation](https://machinelearningmastery.com/k-fold-cross-validation/)
 - [ ] Exploding and vanishing gradients
 
### Activation Function
 - ReLU vs Leaky ReLU
 - Sigmoid activation
 - [ ] [Activation Functions in NN (Sigmoid, tanh, ReLU, Leaky ReLU)](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
 - [ ] [Softmax]()

### Feature Transformation

 - [ ] [correlation - In supervised learning, why is it bad to have correlated features? - Data Science Stack Exchange](https://datascience.stackexchange.com/questions/24452/in-supervised-learning-why-is-it-bad-to-have-correlated-features)
 - [ ] [5.4 Feature Interaction | Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/interaction.html)
 - [ ] [Feature Transformation for Machine Learning, a Beginners Guide | by Rebecca Vickery | vickdata | Medium](https://medium.com/vickdata/four-feature-types-and-how-to-transform-them-for-machine-learning-8693e1c24e80)
 - [ ] [Feature Transformation. How to handle different feature types… | by Ali Masri | Towards Data Science<](https://towardsdatascience.com/apache-spark-mllib-tutorial-7aba8a1dce6e)

### Python Pandas
 - [ ] [(2) Python Pandas Tutorial (Part 8): Grouping and Aggregating - Analyzing and Exploring Your Data](https://www.youtube.com/watch?v=txMdrV1Ut64)
 - [ ] [(2) Python Pandas Tutorial (Part 2): DataFrame and Series Basics - Selecting Rows and Columns](https://www.youtube.com/watch?v=zmdjNSmRXF4&list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS&index=2)

