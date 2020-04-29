import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

bedroom ={'1': 1, '2':2,'3':3,'4':4,'5':5}
bathrooms = {'1': 1, '2':2,'3':3,'4':4,'5':5}
toilets = {'1': 1, '2':2,'3':3,'4':4,'5':5,'6':6}
parking = {'1': 1, '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
estate = {'Yes':'1', 'No':'0'}
location_rank = {'Gbagada':'gbagada', 'Surulere':'surulere','Ajah':'ajah','Ikeja':'ikeja','Ikorodu':'ikorodu',
'Ikoyi':'ikoyi','IyanaIpaja':'iyanaipaja','Lekki':'lekki', 'Ogba':'ogba', 'Yaba':'yaba'}
terraced = {'Yes':'1', 'No':'0'}
New_flag = {'Yes':'1', 'No':'0'}
exec_flag = {'Low 1': '1', 'Lower Middle 2':'2','Middle Range 3':'3',' Highbrow 4':'4'}
serviced_flag = {'Yes':'1', 'No':'0'}

#Get the keys in dictionary
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return str(value)

#Find the keys in the dictionary
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

#Load Model
# def load_pred(model_file):
#     loaded_model =  joblib.load("m")
modelz = joblib.load("modelrf23.pkl")

def main():
    """Housing Ml App"""
    st.title("Housing Pricing App")
    st.subheader("Built with Streamlit") 

    #menu
    menu = ["Prediction", "About"]
    choices = st.sidebar.selectbox("Select Activities",menu)

    if choices == 'Prediction':
        st.subheader("Prediction")
        Number_of_Bedrooms = st.sidebar.slider("Number of Bedrooms", 1,5)
        parking_space = st.sidebar.selectbox("Number of Parking Space",tuple(parking.keys()) )      
        Number_of_Bathrooms = st.sidebar.slider("Number of Bathrooms", 1,9)
        Number_of_Toilet = st.sidebar.selectbox("Number of Toilet",tuple(toilets.keys()) )
        estate_or_not = st.sidebar.selectbox("Do you want to live in an Estate",tuple(estate.keys()) )
        locations = st.sidebar.selectbox("Your preferred location",tuple(location_rank.keys()) )
        terrace_or_not = st.sidebar.selectbox("Do you want to live in an Terracced Apartment",tuple(terraced.keys()) )
        Number_of_flag = st.sidebar.selectbox("Do you prefer a new apartment",tuple(New_flag.keys()) )
        exec_flagg1 = st.sidebar.selectbox("What type of apartment do you want ?",tuple(exec_flag.keys()) )
        serviced_flag1 = st.sidebar.selectbox("Do you prefer an serviced apartment",tuple(serviced_flag.keys()) )

        #Encoding
        v_estate_or_not = get_value(estate_or_not,estate )
        v_locations = get_value(locations,location_rank )        
        v_terrace_or_not = get_value(terrace_or_not,terraced )
        v_Number_of_flag = get_value(Number_of_flag,New_flag )
        v_exec_flagg1 = get_value(exec_flagg1,exec_flag )
        v_serviced_flag1 = get_value(serviced_flag1,serviced_flag )

        #Function to convert Cleaned data csv and Location rank csv to Panda dataframe
        data = load_data('clean_data.csv')
        data2 =load_data('locationrank.csv')

        #Block of code that execute when you click evaluate
        if st.button("Evaluate"):
            #Join Location and Number of bedrooms to produce locationbed variable
            locationbed= v_locations + str(Number_of_Bedrooms)
            df1 = data2[(data2['locationbed'] == locationbed)  ]
            locationBedRank = df1.location_rank.values[0]
            
            #Arranging predictor data in the same way model was trained
            predictor_data= [Number_of_Bedrooms,Number_of_Bathrooms,Number_of_Toilet,v_estate_or_not,locationBedRank,
            v_terrace_or_not, v_Number_of_flag, v_exec_flagg1,v_serviced_flag1]
            predictor_data= np.array(predictor_data).reshape(1,-1)
            z= data.where(data['location']==v_locations)
            b= z.new_price.max()
            c = z.new_price.min()
            d = int(c)
            e = int(b)
            data['price'][(data['location']=="gbagada")&(data['bedrooms']==Number_of_Bathrooms)].plot(kind="hist")
            st.pyplot()
            
            predicted = modelz.predict(predictor_data)
            predicted =  int(predicted)             
            st.write("The predicted price for this apartment in ", locations,"is",predicted)
            st.write("Maximum amount for this apartment type in ",locations, "is", e)
            st.write("Minimum amount for this apartment type in ",locations, "is", d)

    if choices == 'About':
        st.subheader("About")



if __name__ == "__main__":
    main()