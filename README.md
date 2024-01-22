SapiensKNN (K-Nearest Neighbors) is an algorithm for classification and regression that returns the result based on the Euclidean distance between the input values.

# SapiensKNN

The SapiensKNN or Sapiens for K-Nearest Neighbors is a Machine Learning algorithm focused on data classification, where the response for each input is calculated based on the smallest Euclidean distance between the prediction input and the training inputs. The returned value for classification will always be one of the labels from the learning DataSet. If the value of the parameter K is greater than 1, the class that is most repeated among the nearest neighbors represented in K will be returned. Although the algorithm's primary focus is on data classification, it can also potentially be used for regression by returning the average of the values of the selected neighbors with the parameter K.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install SapiensKNN.

```bash
pip install sapiensknn
```

## Usage
Basic usage example:
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN() # class object instantiation
# model training for learning assimilation
inputs = [1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60, 7, 70, 8, 80, 9, 90] # input examples
outputs = ['unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten'] # output examples
sapiensknn.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [15, 1, 25, 2, 35, 3, 45, 4, 55, 5, 65, 6, 75, 7, 85, 8, 95, 9] # inputs to be predicted
results = sapiensknn.predict(inputs=inputs) # calling the prediction function that will return the results
print(results) # displays predicted results
```
```bash
['ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit']
```
You can also define a training matrix, where each vector in the matrix will represent an input.
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN() # class object instantiation
# model training for learning assimilation
inputs = [[1, 2], [10, 20], [3, 4], [30, 40], [5, 6], [50, 60], [7, 8], [70, 80]] # two-dimensional matrix with the input examples
outputs = ['units', 'tens', 'units', 'tens', 'units', 'tens', 'units', 'tens'] # output examples
sapiensknn.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [[2, 3], [20, 30], [4, 5], [40, 50], [6, 7], [60, 70], [8, 9], [80, 90]] # the inputs to be predicted must have the same dimensionality as the learning inputs
results = sapiensknn.predict(inputs=inputs) # calling the prediction function that will return the results
print(results) # displays predicted results
```
```bash
['units', 'tens', 'units', 'tens', 'units', 'tens', 'units', 'tens']
```
The input parameter only accepts vectors or matrices, but the output parameter accepts arrays with any type of dimensionality.
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN() # class object instantiation
# model training for learning assimilation
inputs = [[1, 2], [10, 20], [3, 4], [30, 40], [5, 6], [50, 60], [7, 8], [70, 80]] # two-dimensional matrix with the input examples
outputs = [['units'], ['tens'], ['units'], ['tens'], ['units'], ['tens'], ['units'], ['tens']] # the output elements can be either scalar values or arrays of any dimensionality.
sapiensknn.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [[2, 3], [20, 30], [4, 5], [40, 50], [6, 7], [60, 70], [8, 9], [80, 90]] # the inputs to be predicted must have the same dimensionality as the learning inputs
results = sapiensknn.predict(inputs=inputs) # calling the prediction function that will return the results
print(results) # displays predicted results
```
```bash
[['units'], ['tens'], ['units'], ['tens'], ['units'], ['tens'], ['units'], ['tens']]
```
You can use scalar values of any type to compose the input and output elements.
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN() # class object instantiation
# model training for learning assimilation
inputs = [[-1, 0, 2.5, 'a', 5j, False], [-10, 20, 37.2, 'b', 12j, True], [-2, 7, 0.1, 'c', -7j, False], [-24, 18, 51.9, 'd', 14j, True]] # example with elements of multiple types
outputs = [0, 1, 0, 1] # classification/labeling of input examples
sapiensknn.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [[-15, 43, 71.4, 'b', 13j, True], [-2, 0, 1.3, 'a', 8j, False], [-54, 19, 67.8, 'd', 22j, True], [-3, 9, 0.3, 'c', -7j, False]] # prediction lists must have the same number of elements as training lists
results = sapiensknn.predict(inputs=inputs) # calling the prediction function that will return the results
print(results) # displays predicted results
```
```bash
[1, 0, 1, 0]
```
In the constructor parameters you can configure the number of nearest neighbors to be considered, the normalization of the input data and the type of predictive response (classification or regression).
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN( # class object instantiation
    k=3, # the parameter variable "k", controls the total number of closest neighbors of the input, which will be selected in order of Euclidean proximity (default 1)
    normalization=True, # the "normalization" parameter variable will normalize the elements to keep them on the same proportion scale so that no column has more weight than another. If it is set to True it will normalize, if it is set to False it will keep the elements with their original values (default False)
    regression=False # if the "regression" parameter variable is set to True, the result will return a regression calculation based on the average of the values of the nearest neighbors selected by "k", if it is set to False, the result will return the classification of the prediction inputs (default False)
) # if "k" is greater than one, the classification result will be the class that is most repeated among the closest neighbors, otherwise the classification result will be the output of the training input closest to the prediction input
# model training for learning assimilation
inputs = [[-1, 0, 2.5, 'a', 5j, False], [-10, 20, 37.2, 'b', 12j, None], [-2, 7, 0.1, 'c', -7j, False], [-24, 18, 51.9, 'd', 14j, True]] # elements of type None can also be used without any type of limitation
outputs = [0, 1, 0, 1] # classification/labeling of input examples
sapiensknn.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [[-15, 43, 71.4, 'b', 13j, None], [-2, 0, 1.3, 'a', 8j, False], [-54, 19, 67.8, 'd', 22j, True], [-3, 9, 0.3, 'c', -7j, None]] # prediction lists must have the same number of elements as training lists
results = sapiensknn.predict(inputs=inputs) # calling the prediction function that will return the results
print(results) # displays predicted results
```
```bash
[1, 0, 1, 0]
```
To obtain a regressive result you must assign a value greater than 1 to the "k" parameter and set True to the "regression" parameter. In regression calculations, the training output list must contain only scalar values or one-dimensional vectors.
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN( # class object instantiation
    k=2, # the parameter variable "k", controls the total number of closest neighbors of the input, which will be selected in order of Euclidean proximity (default 1)
    normalization=False, # the "normalization" parameter variable will normalize the elements to keep them on the same proportion scale so that no column has more weight than another. If it is set to True it will normalize, if it is set to False it will keep the elements with their original values (default False)
    regression=True # if the "regression" parameter variable is set to True, the result will return a regression calculation based on the average of the values of the nearest neighbors selected by "k", if it is set to False, the result will return the classification of the prediction inputs (default False)
) # use a value greater than 1 in "k" to enable the regression parameter
# training model for learning assimilation with simple regression
inputs = [2, 4, 6, 8, 10] # example of independent variables
outputs = [1, 2, 3, 4, 5] # example of dependent variables
sapiensknn.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [3, 5, 7, 9] # independent variables to calculate the dependent variables based on training patterns
results = sapiensknn.predict(inputs=inputs) # calling the prediction function that will return the results
print(results) # displays the regression result
```
```bash
[1.5, 2.5, 3.5, 4.5]
```
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN( # class object instantiation
    k=4, # the parameter variable "k", controls the total number of closest neighbors of the input, which will be selected in order of Euclidean proximity (default 1)
    normalization=False, # the "normalization" parameter variable will normalize the elements to keep them on the same proportion scale so that no column has more weight than another. If it is set to True it will normalize, if it is set to False it will keep the elements with their original values (default False)
    regression=True # if the "regression" parameter variable is set to True, the result will return a regression calculation based on the average of the values of the nearest neighbors selected by "k", if it is set to False, the result will return the classification of the prediction inputs (default False)
) # use a value greater than 1 in "k" to enable the regression parameter
# training model for learning assimilation with multivariable regression
inputs = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5], [7, 8], [8, 7], [9, 10], [10, 9]] # example of independent variables
outputs = [3, 3, 7, 7, 11, 11, 15, 15, 19, 19] # example of dependent variables
sapiensknn.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [[2, 3], [4, 5], [6, 7], [8, 9]] # independent variables to calculate the dependent variables based on training patterns
results = sapiensknn.predict(inputs=inputs) # calling the prediction function that will return the results
print(results) # displays the regression result
```
```bash
[5.0, 9.0, 13.0, 17.0]
```
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN( # class object instantiation
    k=4, # the parameter variable "k", controls the total number of closest neighbors of the input, which will be selected in order of Euclidean proximity (default 1)
    normalization=False, # the "normalization" parameter variable will normalize the elements to keep them on the same proportion scale so that no column has more weight than another. If it is set to True it will normalize, if it is set to False it will keep the elements with their original values (default False)
    regression=True # if the "regression" parameter variable is set to True, the result will return a regression calculation based on the average of the values of the nearest neighbors selected by "k", if it is set to False, the result will return the classification of the prediction inputs (default False)
) # use a value greater than 1 in "k" to enable the regression parameter
# training model for learning assimilation with multivariate regression
inputs = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5], [7, 8], [8, 7], [9, 10], [10, 9]] # example of independent variables
outputs = [[3, 30], [3, 30], [7, 70], [7, 70], [11, 110], [11, 110], [15, 150], [15, 150], [19, 190], [19, 190]] # example of dependent variables
sapiensknn.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [[2, 3], [4, 5], [6, 7], [8, 9]] # independent variables to calculate the dependent variables based on training patterns
results = sapiensknn.predict(inputs=inputs) # calling the prediction function that will return the results
print(results) # displays the regression result
```
```bash
[[5.0, 50.0], [9.0, 90.0], [13.0, 130.0], [17.0, 170.0]]
```
To save a pre-trained model you must call the "saveModel" method. This method receives in the parameter called "path" a string with the path and name of the file to be saved. If no value is assigned to the "path" parameter, the file will be saved with a default name.
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN() # class object instantiation
# training phase (pattern recognition)
inputs = [[1, 2, 3], [10, 20, 30], [100, 200, 300], [4, 5, 6], [40, 50, 60], [400, 500, 600]] # input examples
outputs = ['units', 'tens', 'hundreds', 'units', 'tens', 'hundreds'] # output examples
sapiensknn.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
sapiensknn.saveModel(path='my_model') # saves the pre-trained model with the prefix "my_model"
# in this case the file will be saved in the current directory because no path was defined before the name
# the file will be saved with the name "my_model-32P.knn", that is, the prefix "my_model" defined in "path", plus the number of model parameters followed by the ".knn" extension
# you can rename the saved file to any name you prefer as long as the ".knn" extension is maintained
```
To load a pre-trained model without the need to train it again, simply call the "loadModel" method, which will receive the address and file name of the model to be loaded in the "path" parameter.
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN() # class object instantiation
# prediction phase or execution phase (application of knowledge)
sapiensknn.loadModel(path='my_model-32P.knn') # loads the pre-trained model
inputs = [[2, 3, 4], [20, 30, 40], [200, 300, 400], [3, 4, 5], [30, 40, 50], [300, 400, 500]] # values to be predicted (may be the same or different from the training input values, as long as they respect the same standard)
results = sapiensknn.predict(inputs=inputs) # calling the prediction function that will return the results
print(results) # displays the regression result
```
```bash
['units', 'tens', 'hundreds', 'units', 'tens', 'hundreds']
```
Constructor parameters do not affect training, only prediction. So you can use different construction configurations in different predictions for the same model.
```python
from sapiensknn import SapiensKNN # module main class import
# if the calculation returns the same number of occurrences for different classes, only the closest occurrence will be considered
sapiensknn = SapiensKNN(k=2) # class object instantiation (k equals 2 for current forecast)
# prediction phase or execution phase (application of knowledge)
sapiensknn.loadModel(path='my_model-32P.knn') # loads the pre-trained model
inputs = [[2, 3, 4], [20, 30, 40], [200, 300, 400], [3, 4, 5], [30, 40, 50], [300, 400, 500]] # values to be predicted (may be the same or different from the training input values, as long as they respect the same standard)
results = sapiensknn.predict(inputs=inputs) # calling the prediction function that will return the results
print(results) # displays the regression result
```
```bash
['units', 'tens', 'hundreds', 'units', 'tens', 'hundreds']
```
It is also possible to transfer learning from one model to another using the "transferLearning" method. This method receives three parameters, in the first parameter we define the path to the model that will transfer the learning, in the second parameter we define the path to the model that will receive the learning and in the third parameter we define the path to the model that will be saved with the union of learning from the two previous models.
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN() # class object instantiation
# training phase (pattern recognition)
inputs = [[1, 2, 3], [10, 20, 30], [4, 5, 6], [40, 50, 60]] # input examples
outputs = ['units', 'tens', 'units', 'tens'] # output examples
sapiensknn.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
sapiensknn.saveModel(path='transmitter_model') # saves the pre-trained model with the prefix "transmitter_model"
```
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN() # class object instantiation
# training phase (pattern recognition)
inputs = [[70, 80, 90], [700, 800, 900]] # input examples
outputs = ['tens', 'hundreds'] # output examples
sapiensknn.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
sapiensknn.saveModel(path='receiver_model') # saves the pre-trained model with the prefix "receiver_model"
```
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN() # class object instantiation
# learning transfer phase
sapiensknn.transferLearning( # method for combining learnings
    transmitter_path='transmitter_model-24P.knn', # model that will transmit learning
    receiver_path='receiver_model-16P.knn', # model that will receive the learning
    rescue_path='complete_model' # model that will be saved with the complete union of learnings
) # the model will be saved with the name "complete_model-32P.knn"
# the final model may have fewer parameters than the sum of the previous parameters because there are some configuration parameters that will be the same and will not need repetition
```
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN() # class object instantiation
# prediction phase or execution phase (application of knowledge)
sapiensknn.loadModel(path='complete_model-32P.knn') # loads the pre-trained model
inputs = [[2, 3, 4], [20, 30, 40], [600, 700, 800]] # values to be predicted
results = sapiensknn.predict(inputs=inputs) # calling the prediction function that will return the results
print(results) # displays the regression result
```
```bash
['units', 'tens', 'hundreds']
```
You can use the "test" function to test your model's learning. This function will return a data dictionary with the percentage of hits in the "hits" key and the percentage of errors in the "errors" key. If the level of assertiveness is not meeting your needs, you can retrain your model with a greater amount of example data and/or with a greater variability of input values.
```python
from sapiensknn import SapiensKNN # module main class import
sapiensknn = SapiensKNN() # class object instantiation
# test phase (testing the level of learning)
sapiensknn.loadModel(path='complete_model-32P.knn') # loads the pre-trained model
inputs = [[2, 3, 4], [20, 30, 40], [200, 300, 400], [3, 4, 5], [30, 40, 50], [300, 400, 500], [4, 5, 6], [40, 50, 60], [400, 500, 600], [5, 6, 7], [50, 60, 70], [500, 600, 700]] # values to be predicted
outputs = ['units', 'tens', 'hundreds', 'units', 'tens', 'hundreds', 'units', 'tens', 'hundreds', 'units', 'tens', 'hundreds'] # expected values as a response in prediction
results = sapiensknn.test(inputs=inputs, outputs=outputs) # function call to test learning
print(results) # displays the test result with the percentage of correct answers and the percentage of errors with values between 0 and 1 (0 for 0% and 1 for 100%)
# the acceptable level of assertiveness will depend on the project's precision needs, but for most cases we use the pareto rule (up to 20% errors are tolerable)
```
```bash
{'hits': 0.8333333333333334, 'errors': 0.16666666666666663}
```

## Methods
### Construtor: SapiensKNN

Parameters
| Name          | Description                       | Type | Default Value |
|---------------|-----------------------------------|------|---------------|
| k             | Number of nearest neighbors       | int  | 1             |
| normalization | Enable or disable data normalization | bool | False         |
| regression    | Enable or disable regression calculation | bool | False         |

### fit: Train the model with input and output examples for pattern recognition
Parameters
| Name    | Description                                     | Type  | Default Value |
|---------|-------------------------------------------------|-------|---------------|
| inputs  | Input list for training with scalar or vector values | list  | []            |
| outputs | Output list for training with scalar, vector, matrix or tensor values | list  | []            |

### saveModel: Saves a file with the current model training
Parameters
| Name | Description                                       | Type | Default Value |
|------|---------------------------------------------------|------|---------------|
| path | Path with the address and file name of the model to be saved | str  | ''           |

### loadModel: Load a pre-trained model
Parameters
| Name | Description                                       | Type | Default Value |
|------|---------------------------------------------------|------|---------------|
| path | Path with the address and file name of the model to be loaded | str  | ''            |

### transferLearning: Transfer learning from one model to another
Parameters
| Name             | Description                                               | Type | Default Value |
|------------------|-----------------------------------------------------------|------|---------------|
| transmitter_path | Path with the address and file name of the model that will transfer the learning | str  | ''            |
| receiver_path    | Path with the address and file name of the model that will receive the learning | str  | ''            |
| rescue_path      | Path with address and name of the model file that will be saved with both learnings | str  | ''            |

### predict: Returns the result list with the predicted values
Parameters
| Name   | Description                                   | Type | Default Value |
|--------|-----------------------------------------------|------|---------------|
| inputs | Input list for prediction with scalar or vector values | list | []            |

### test: Tests learning by returning a dictionary with the percentage of hits and errors
Parameters
| Name   | Description                                       | Type  | Default Value |
|--------|---------------------------------------------------|-------|---------------|
| inputs | Input list for prediction with scalar or vector values | list  | []            |
| outputs| Test output list with scalar, vector, matrix or tensor values that are expected as a response | list  | []            |

Check out examples below with real data using the Titanic DataSet and Iris DataSet databases.
```bash
pip install pandas pyarrow
```
## Example with data from the Titanic DataSet
```python
from pandas import read_csv # import csv file reading module with pandas
from sapiensknn import SapiensKNN # import of sapiensknn module main class
sapiensknn = SapiensKNN(normalization=True) # class instantiation with data normalization
# data preparation phase for training and testing
data_df = read_csv('titanic.csv') # reading csv file converted to dataframe
# selection of input and output lists
input_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'] # naming the columns that will make up the input list
inputs = data_df[input_columns].values.tolist() # reading input column values converted into lists
outputs = data_df['Survived'].values.tolist() # reading the column containing the output labels (the result will be a list with the label values)
# separation of data between training and testing
fit_x, fit_y = inputs[:int(len(inputs)*.8)], outputs[:int(len(outputs)*.8)] # separates 80% of input and output data for training
test_x, test_y = inputs[int(len(inputs)*.8):], outputs[int(len(outputs)*.8):] # separates the remaining 20% of data for testing
# machine learning model training phase
sapiensknn.fit(inputs=fit_x, outputs=fit_y) # performs machine learning model training
sapiensknn.saveModel() # saves the machine learning model
# machine learning model testing phase
results = sapiensknn.test(inputs=test_x, outputs=test_y) # tests the learning of the trained model
print(results) # displays percentage test results
```
```bash
{'hits': 0.6424581005586593, 'errors': 0.35754189944134074}
```
## Example with data from the Iris DataSet
```python
from pandas import read_csv # import csv file reading module with pandas
from sapiensknn import SapiensKNN # import of sapiensknn module main class
sapiensknn = SapiensKNN(normalization=True) # class instantiation with data normalization
# data preparation phase for training and testing
data_df = read_csv('iris.csv') # reading csv file converted to dataframe
# selection of input and output lists
input_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width'] # naming the columns that will make up the input list
inputs = data_df[input_columns].values.tolist() # reading input column values converted into lists
outputs = data_df['variety'].values.tolist() # reading the column containing the output labels (the result will be a list with the label values)
# separation of data between training and testing
fit_x, fit_y = inputs[:int(len(inputs)*.8)], outputs[:int(len(outputs)*.8)] # separates 80% of input and output data for training
test_x, test_y = inputs[int(len(inputs)*.8):], outputs[int(len(outputs)*.8):] # separates the remaining 20% of data for testing
# machine learning model training phase
sapiensknn.fit(inputs=fit_x, outputs=fit_y) # performs machine learning model training
sapiensknn.saveModel() # saves the machine learning model
# machine learning model testing phase
results = sapiensknn.test(inputs=test_x, outputs=test_y) # tests the learning of the trained model
print(results) # displays percentage test results
```
```bash
{'hits': 0.8, 'errors': 0.19999999999999996}
```

## Contributing

We do not accept contributions that may result in changing the original code.

Make sure you are using the appropriate version.

## License

This is proprietary software and its alteration and/or distribution without the developer's authorization is not permitted.
