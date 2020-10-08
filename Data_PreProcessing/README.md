# Data PreProcessing Tools
---


## What is Data PreProcessing ?

In any Machine Learning process, Data Preprocessing is that step in which the data gets transformed, or Encoded, to bring it to such a state that now the machine can easily parse it. In other words, the features of the data can now be easily interpreted by the algorithm.


## What is Features in Machine Learning ?

Data objects are described by a number of features, that capture the basic characteristics of an object, such as the mass of a physical object or the time at which an event occurred, etc.. Features are often called as variables, characteristics, fields, attributes, or dimensions.


### A feature is an individual measurable property or characteristic of a phenomenon being observed

![data_preprocessing](data_preprocessing.jpg)


Features can be:

* Categorical : Features whose values are taken from a defined set of values. For instance, days in a week : {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday} is a category because its value is always taken from this set. Another example could be the Boolean set : {True, False}

* Numerical : Features whose values are continuous or integer-valued. They are represented by numbers and possess most of the properties of numbers. For instance, number of steps you walk in a day, or the speed at which you are driving your car at.
---


### Step 1: Import Libraries
First step is usually importing the libraries that will be needed in the program. A library is essentially a collection of modules that can be called and used. 

### Step 2: Import the Dataset
A lot of datasets come in CSV formats. We will need to locate the directory of the CSV file at first and read it using a method called read_csv which can be found in the library called pandas.

### Step 3: Taking care of Missing Data in Dataset
Sometimes you may find some data are missing in the dataset. We need to be equipped to handle the problem when we come across them.

### Step 4: Encoding categorical data
Sometimes our data is in qualitative form, that is we have texts as our data. We can find categories in text form. Now it gets complicated for machines to understand texts and process them, rather than numbers, since the models are based on mathematical equations and calculations. Therefore, we have to encode the categorical data.

### Step 5: Splitting the Dataset into Training set and Test Set
Now we need to split our dataset into two sets — a Training set and a Test set. We will train our machine learning models on our training set, i.e our machine learning models will try to understand any correlations in our training set and then we will test the models on our test set to check how accurately it can predict.

### Step 6: Feature Scaling
The final step of data preprocessing is to apply the very important feature scaling.

### What is Feature Scaling
It is a method used to standardize the range of independent variables or features of data.

### Why is it necessary? 
A lot of machine learning models are based on Euclidean distance. If, for example, the values in one column (x) is much higher than the value in another column (y), (x2-x1) squared will give a far greater value than (y2-y1) squared. So clearly, one square difference dominates over the other square difference. In the machine learning equations, the square difference with the lower value in comparison to the far greater value will almost be treated as if it does not exist. We do not want that to happen. That is why it is necessary to transform all our variables into the same scale. 

![data_preprocessing](data_preprocessingimg2.jpg)