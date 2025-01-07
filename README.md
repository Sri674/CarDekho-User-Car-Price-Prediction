CARDEKHO: User Car Price Prediction

ABOUT: CarDekho is an online automotive marketplace that helps users buy, sell, and research cars in India.

OBJECTIVE: My aim is to enhance the customer experience and streamline the pricing process by leveraging machine learning. I have created an accurate and user-friendly streamlit tool that predicts the prices of used cars based on various features. This tool should be deployed as an interactive web application for both customers and sales representatives to use seamlessly.

DOCUMENTATION(METHODOLOGY):
                        
                        STEP 1) By received Unstructed dataset,i have converted that to Structed Format.The dataset provided of cities namely Chennai, Bangalore, Delhi, Kolkata, Hyderabad, Jaipur is in list of dictionary form, i have converted that to DataFrame by various techniques like looping,ast etc(ast=Abstract Syntax Tree), finally combined all cities dataset into single dataset(by using concat)[unstruct to struct file]
                       
                        Step 2) Data cleaning(changing datatype, removing null values, removing symbols, removed unwanted columns). In this i have handed null values in Categorical and Numerical columns. For numerical columns, use techniques like mean, median, or mode imputation. For categorical columns, use mode imputation or create a new category for missing values. (Mean- When data is Normally Distributed, Median- When Data is Skewed, Mode-appears most frequently in the dataset. [Data cleaning and preprocessing file]
                        
                        Step 3) Used LabelEncoder for Categorical columns, to convert into numerical values(Because M.L always learns numerical values).In this certain columns like engine, i have treated that as categorical columns, and i have categorized that and encode that column values. I have also changed Column like Year of Manufacture to Car Age, will be easy for the M.L model for its better performance(df['Car_Age'] = current_year - df['Year of Manufacture']). [Data cleaning and preprocessing file]
                        
                        Step 4) Removes the Outliers(By setting upper_limit and lower_limit), and followed by that i have done StandardScaler for numerical columns.[Data cleaning and preprocessing file]
                        
                        Step 5) Done EDA Analysis by Various methods namely Descriptive Statistics(Mean, median, mode, standard deviation), Data Visualization(done scatter plots, histograms, box plots, and correlation heatmaps), and Feature Selection(By using RandomForestRegressor, Found out the important feature in the column that heavily influence price).[EDA Analysis file]
                       
                        Step 6) MODEL DEVELOPMENT: By assinging features(remaining columns) and target(price) and doing train-test-split, i have performed various models like linear regression, DecisionTree, RandomForest, and GradientBoostingMachines and their evaluation metrics like MAE,MSE,R2_Score.I also done hyperparametertuning for each model using GridSearchCV to increase model performance.By comparing every model performance(by seeing Evaluation metrics), i choosed best performance model(RandomForestRegressor) for deployment in streamlit. i also checked training accuracy and testing accuracy, whether there is any overfitting or not.[Model development file]
                        Step 7) Deployed the final model using Streamlit to create an interactive web application. Allow users to input car features and get real-time price predictions. Ensure the application is user-friendly and intuitive.[carprediction.py file]

CONCLUSION: The main motive of this project is to give user-friendly interactive web application for customers to get real-time price prediction.(the main role plays here to showcase the price is M.L (RandomForestRegressor))



      
