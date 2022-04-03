'''
this is the code for the thyroid detection web app using streamlit.

'''

# Importing important libraries
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date
import pickle


# Loading the pickle file here
pickle_in = open("car_price_prediction.pkl","rb")
predictor = pickle.load(pickle_in)

#Date time
today = date.today()
f = open("file_logger.txt", "a")


def main():
    
    
    # Code for Heading
    html_temp = """
   <div style="background-color:tomato;padding:10px">
   <h2 style="color:white;text-align:center;">Streamlit Used Car Price Prediction ML App </h2>
   </div>
   """
   
   #Code for heading
    st.markdown(html_temp, unsafe_allow_html=True)
    st.info("")
    
    #--------------------------------------------------------------------------
    
    #Code for traning file uploader for model in app.
    data_train = st.file_uploader("Choose a csv file to train Model", ["csv"])
    
    
    if data_train is not None:
        f.write(f"{today} : Dataset uploaded")
        f.write("\n")
        
        df = pd.read_csv(data_train)
        st.markdown("Dataset you have uploaded:-")
        st.dataframe(df)
        
        f.write(f"{today} : Dataset Shown")
        f.write("\n")
        
        try:
            
            f.write(f"{today} : Traning Started")
            f.write("\n")
            
            f.write(f"{today} : preprocessing for traning data Started!")
            f.write("\n")

            df['Current_Year'] = 2021
            
            df['No_of_years'] = df['Current_Year'] - df['Year']
            
            df.drop(['Year'], axis=1, inplace=True)
            
            df.drop(['Current_Year'], axis=1, inplace=True)
            
            df.drop(['Car_Name'], axis=1, inplace=True)

            Fuel = df['Fuel_Type']
            Fuel = pd.get_dummies(Fuel, drop_first=True)

            Seller = df['Seller_Type']
            Seller = pd.get_dummies(Seller, drop_first=True)

            User = df['Transmission']
            User = pd.get_dummies(User, drop_first=True)
            
            final_dataset = pd.concat([df,Fuel,Seller,User], axis=1)
                    
            final_dataset.drop(['Fuel_Type','Seller_Type','Transmission'], axis=1, inplace=True)
            
            f.write(f"{today} : preprocessing for traning data Complete!")
            f.write("\n")
            
            X = final_dataset.iloc[:,1:9] 
            y = final_dataset.iloc[:,0] 
            
            f.write(f"{today} : Spliting into traning and testing data with testing size as 0.20")
            f.write("\n")
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            
            f.write(f"{today} : pridiction started on traning data")
            f.write("\n")
            
            predictor.fit(X_train, y_train)
            
            
            predictions = predictor.predict(X_test)
            
            f.write(f"{today} : pridiction on traning data complete!")
            f.write("\n")
            
            from sklearn import metrics
            MAE =  metrics.mean_absolute_error(y_test, predictions)
            MSE =  metrics.mean_squared_error(y_test, predictions)
            RMSE =  np.sqrt(metrics.mean_squared_error(y_test, predictions))
            
            f.write(f"{today} : MAE : {MAE}, MSE : {MSE},  RMSE : {RMSE}")
            f.write("\n")

            
            st.success("Traning Complete")
            
            f.write(f"{today} : Traning Complete")
            f.write("\n")
            
        
        except Exception as e:
            st.error("Invalid Formate of Dataset, Please! Try Again")
            
            f.write(f"{today} : {e}")
            f.write("\n")
            
        
    #-------------------------------------------------------------------------
        
        
        
        
    #Code for uploding the file to predict in app
    data = st.file_uploader("Choose a csv file for Prediction", ["csv"])
    
    if data is not None:
        
        f.write(f"{today} : Dataset uploaded")
        f.write("\n")
        
        df = pd.read_csv(data)
        st.markdown("Dataset you have uploaded:-")
        st.dataframe(df)
        
        f.write(f"{today} : Dataset Shown")
        f.write("\n")
        
        try:
            
            f.write(f"{today} : Prediction Started")
            f.write("\n")
            
            f.write(f"{today} : Preprocessing for test data Started")
            f.write("\n")

            df['Current_Year'] = 2021
            
            df['No_of_years'] = df['Current_Year'] - df['Year']
            
            df.drop(['Year'], axis=1, inplace=True)
            
            df.drop(['Current_Year'], axis=1, inplace=True)
            
            df.drop(['Car_Name'], axis=1, inplace=True)

            Fuel = df['Fuel_Type']
            Fuel = pd.get_dummies(Fuel, drop_first=True)

            Seller = df['Seller_Type']
            Seller = pd.get_dummies(Seller, drop_first=True)

            User = df['Transmission']
            User = pd.get_dummies(User, drop_first=True)

            final_dataset = pd.concat([df,Fuel,Seller,User], axis=1)

            final_dataset.drop(['Fuel_Type','Seller_Type','Transmission'], axis=1, inplace=True)
            
            f.write(f"{today} : Preprocessing for test data complete")
            f.write("\n")
        
        
            ans = predictor.predict(final_dataset)
            df["predcited_value"] = ans
            
            f.write(f"{today} : Prediction Complete")
            f.write("\n")
            
            st.markdown("Dataset after Prediction:-")
            st.dataframe(df)
            
            f.write(f"{today} : Dataset Shown after prediction")
            f.write("\n")
            
            if st.button("Download"):
                df.to_csv("Result.csv")
                st.success("Download Complete")
                
                f.write(f"{today} : Dataset Downloaded after prediction")
                f.write("\n")
                
        
           
        except Exception as e:
            st.error("Invalid Formate of Dataset, Please! try Again")
            
            f.write(f"{today} : {e}")
            f.write("\n")
            
     
    #--------------------------------------------------------------------------
        
        
        
    #Code for Or Markdown in App 
    html_temp = """
    <h2 style="color:White;text-align:center;">Or</h2>
    </div>
    """

    #Code for Or Markdown
    st.markdown(html_temp,unsafe_allow_html=True)
     
    
    
    
    #Code for Prediction using parameters given by users
    Present_Price = st.text_input("Present_Price:- for ex (3.2) i.e. 3.2lac")
    Kms_Driven = st.text_input("Kms_Driven:- ")
    Owner = st.selectbox("Owner:- ", [0, 1, 2, 3])
    No_of_years = st.text_input("No_of_years:- ")
    Diesel = st.selectbox("Diseal:- 1 for yes, 0 for No", [0, 1])
    Petrol = st.selectbox("Petrol:- 1 for yes, 0 for No", [0, 1])
    Individual = st.selectbox("Individual:- 1 for yes, 0 for No", [0, 1])
    Manual = st.selectbox("Manual:- 1 for yes, 0 for No", [0, 1])

    
    
    #Code button which predicts the output 
    if st.button("Predict"):
        
        try:
            f.write(f"{today} : Prediction for individual Started")
            f.write("\n")
        
            result = predictor.predict([[Present_Price,Kms_Driven,Owner,
                            No_of_years,Diesel,Petrol,Individual,Manual]])
            
            st.markdown("The output will shown in lac for ex 7.2 i.e. 7.2 lac")

            st.success('Predicted Selling Price is {}'.format(result[0]))
            
            f.write(f"{today} : Individual Prediction Complete")
            f.write("\n")
        
        except Exception as e:
            st.error("Invalid Inputs, Please! Try Again")
            
            f.write(f"{today} : {e}")
            f.write("\n")

    
if __name__ == '__main__' :
    main()
    
