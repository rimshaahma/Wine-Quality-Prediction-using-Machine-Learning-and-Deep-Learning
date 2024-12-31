
# **Wine Quality Prediction using Machine Learning and Deep Learning**

## **Project Overview**
The project is focused on predicting wine quality based on physicochemical features using **machine learning** and **deep learning** techniques. The model is trained and evaluated using data collected on the attributes of wine, and the goal is to predict the quality of wine (a numerical value between 0-10) based on features such as acidity, alcohol content, pH, and others.

The project workflow includes:
1. **Data Collection**: Downloading and understanding the dataset.
2. **Data Preprocessing**: Cleaning, handling missing values, and feature scaling.
3. **Model Development**: Building and training a machine learning model.
4. **Model Evaluation**: Assessing model performance using appropriate evaluation metrics.
5. **Deployment**: Deploying the model using a **Flask API** for real-world applications.

The purpose of this project is to demonstrate a comprehensive **end-to-end machine learning pipeline**.

---

## **Dataset Description**

The **Wine Quality Dataset** consists of various physicochemical attributes of wines and their associated quality score. The dataset includes the following columns:

1. **Fixed Acidity**: Amount of non-volatile acids in wine, which contributes to the overall taste profile.
2. **Volatile Acidity**: A measure of volatile acids (e.g., acetic acid) that can negatively impact taste. Higher values can result in an unpleasant taste (vinegar-like).
3. **Citric Acid**: Contributes to the tartness and freshness of the wine.
4. **Residual Sugar**: Sugar content remaining after fermentation. Higher values can make the wine sweeter.
5. **Chlorides**: Salt content in the wine. High levels can influence flavor.
6. **Free Sulfur Dioxide**: SO2 not bound to other compounds, protects the wine from oxidation and microbial growth.
7. **Total Sulfur Dioxide**: Sum of free and bound sulfur dioxide.
8. **Density**: The density of the wine liquid, which is influenced by alcohol and sugar content.
9. **pH**: A measure of acidity or alkalinity. The lower the pH, the more acidic the wine.
10. **Sulphates**: Affects the preservation of the wine.
11. **Alcohol**: Percentage of alcohol by volume in the wine.
12. **Quality**: The target variable; a numerical rating between 0 and 10, with higher values indicating better quality.

---

## **Key Concepts and Definitions**

### **Machine Learning (ML)**:
- A field of computer science that focuses on developing algorithms that allow computers to learn from and make predictions based on data.
- Key types of machine learning:
  - **Supervised Learning**: The model is trained on labeled data. The model learns to map input features to a known output.
  - **Unsupervised Learning**: The model is trained on unlabeled data and tries to find hidden patterns or groupings.
  - **Reinforcement Learning**: The model learns by interacting with an environment and receiving feedback through rewards and penalties.

### **Deep Learning (DL)**:
- A subfield of machine learning that uses neural networks with multiple layers (hence "deep") to model complex patterns in large datasets.
- Deep learning excels at handling unstructured data like images, text, and audio.

### **Regression vs Classification**:
- **Regression**: Predicts continuous values. In this project, predicting the wine quality score is a **regression** problem.
- **Classification**: Predicts discrete class labels (e.g., spam vs non-spam email). If the wine quality were categorized as "Good" or "Bad", it would be a classification problem.

### **Neural Networks**:
- A set of algorithms modeled after the human brain that are used to recognize patterns. They consist of layers of interconnected neurons (nodes).
- Key components of a neural network:
  - **Input Layer**: Accepts the input features (e.g., acidity, alcohol).
  - **Hidden Layers**: Layers between the input and output that perform computations to learn patterns in data.
  - **Output Layer**: Produces the final prediction (wine quality score).
  - **Activation Functions**: Introduces non-linearity, allowing the model to learn more complex patterns (e.g., ReLU, Sigmoid).

### **Random Forest**:
- An ensemble learning algorithm that combines multiple decision trees. It reduces overfitting and increases the model's robustness by averaging the predictions from multiple trees.
- Each decision tree in the random forest is trained on a random subset of the data, and each tree makes a prediction independently.

### **Feature Scaling**:
- The process of standardizing or normalizing the feature values to bring them to the same scale. This is particularly important for models like neural networks that rely on gradient-based optimization.
- **Standardization** (Z-score normalization): Rescaling the data so that it has a mean of 0 and a standard deviation of 1.
- **Min-Max Scaling**: Rescaling the data to fit between a defined range (e.g., [0, 1]).

### **Dropout**:
- A regularization technique used to prevent overfitting in neural networks. During training, a random subset of neurons is "dropped" (set to zero), forcing the network to learn more robust patterns and reduce dependency on specific neurons.

### **Loss Function**:
- A function that calculates the difference between the predicted output and the actual target value. The model aims to minimize this loss function during training.
- **Mean Squared Error (MSE)**: Commonly used in regression tasks. It calculates the average squared difference between predicted and actual values.
- **Mean Absolute Error (MAE)**: Measures the average absolute differences between predicted and actual values.

### **Adam Optimizer**:
- A popular optimization algorithm used for training neural networks. It combines the advantages of two other extensions of stochastic gradient descent (SGD): **AdaGrad** and **RMSProp**. It adjusts the learning rate based on the momentums and gradients of the parameters.

---

## **Steps Involved in the Project**

### **1. Data Preprocessing**

**Purpose**: Preparing the data for training by handling missing values, encoding categorical variables, and scaling the features.

#### Steps:
1. **Load the dataset**: Use the `pandas.read_csv()` method to load the dataset into a DataFrame.
2. **Handle missing values**: Check for and impute any missing values in the dataset.
3. **Feature-target split**: Separate the dataset into features (`X`) and the target variable (`y`).
4. **Feature scaling**: Standardize or normalize the features using `StandardScaler` or `MinMaxScaler` to improve model performance.

---

### **2. Model Development**

**Purpose**: Train a machine learning or deep learning model to predict the wine quality.

#### Deep Neural Network (DNN) Model:
1. **Input Layer**: The number of nodes in the input layer corresponds to the number of features (e.g., 11 features).
2. **Hidden Layers**: Use multiple hidden layers with `ReLU` activation to introduce non-linearity.
3. **Output Layer**: The output layer has 1 neuron (for the predicted wine quality score).
4. **Activation Function**: The hidden layers use **ReLU** (Rectified Linear Unit) to model complex patterns in the data. The output layer does not use any activation for regression.
5. **Loss Function**: **Mean Squared Error (MSE)**, suitable for regression tasks.
6. **Optimizer**: **Adam Optimizer**, an efficient algorithm for training deep neural networks.

#### Random Forest Regressor Model:
1. **Train the model** on the training data.
2. **Predict the results** using the test data.
3. **Evaluate the model** using MSE and MAE.

---

### **3. Model Evaluation**

**Purpose**: Assess the model's ability to generalize on unseen data.

#### Evaluation Metrics:
1. **Mean Squared Error (MSE)**: A loss function used to measure the average squared difference between predicted and actual values.
2. **Mean Absolute Error (MAE)**: The average of the absolute errors between predicted and actual values.

---

### **4. Model Deployment**

**Purpose**: Make the trained model accessible for use in a web application.

#### Steps:
1. **Flask API**: Develop a Flask web application to serve the model. This allows users to send data to the model and receive predictions in real-time.
2. **API Testing**: Use a script (`test_request.py`) to send test data to the deployed API and check the response.

---

## **How to Run the Project**

### **1. Install Dependencies**
Run the following command to install the necessary libraries:
```bash
pip install -r requirements.txt
```

### **2. Train the Model**
Run the Jupyter notebook for model training:
```bash
jupyter notebook model_training.ipynb
```

### **3. Start the Flask API**
To deploy the model via a web service, run the Flask application:
```bash
python app.py
```

### **4. Test the API**
Run the test script to ensure the API is working correctly:
```bash
python test_request.py
```

---

## **Future Work**
1. **Model Tuning**: Experiment with different architectures, hyperparameter tuning, and other regression models (e.g., XGBoost).
2. **Model Deployment**: Explore deployment options like **AWS**, **GCP**, or **Heroku**.
3. **Expand Dataset**: Include more features or use a larger dataset for more robust training.

---

### **Conclusion**
This project demonstrates a full **end-to-end machine learning pipeline** from data loading and preprocessing to model training and deployment. The detailed explanation of each concept and step ensures a solid understanding of how ML and DL models work, and the deployment provides real-world usability.

