import streamlit as st
import pandas as pd
import sqlite3
import pickle
import time
import sklearn
from sklearn.ensemble import RandomForestClassifier

# load file with pickle and read binaruy 
hotelFacilities = pickle.load(open('hotelFacilities.pkl','rb'))
roomFacilities = pickle.load(open('roomFacilities.pkl','rb'))
nearestPoint = pickle.load(open('pointInterests.pkl','rb'))
colOri = pickle.load(open('col.pkl','rb'))

# load model from pikcle 
# xgbModel = pickle.load(open('xgbModel.pkl','rb'))
# svrModel = pickle.load(open('svmModel.pkl','rb'))
# rfModel = pickle.load(open('RFModel.pkl','rb'))

xgbModel = pickle.load(open(r'C:\Users\RESTU\OneDrive - SMA NEGERI 3 SLAWI\kuliah\semester 5\stki\latihan\bengkel koding\model\xgbModel.pkl', 'rb'))
svrModel = pickle.load(open(r'C:\Users\RESTU\OneDrive - SMA NEGERI 3 SLAWI\kuliah\semester 5\stki\latihan\bengkel koding\model\svrModel.pkl','rb'))
rfModel = pickle.load(open(r'C:\Users\RESTU\OneDrive - SMA NEGERI 3 SLAWI\kuliah\semester 5\stki\latihan\bengkel koding\model\RFModel.pkl','rb'))


# mengatur konfigurasi halaman
st.set_page_config(
    page_title = "Estimasi Hotel Yogyakarta",
    page_icon = ':hotel' #nama emoji
)

# st.write dapat digunakan menampilkan test,dataframe,visualisasi
st.title('Yogyakarta Hotel Price Estimation')
st.write('for mor ingo about blablabal')

# st.sidebar dapat digunakan untuk membuat sidebar
st.sidebar.header("User Input Features")

# Input User untuk memasukkan elemen input use dalam sidebar dilakukan
# dengan st.sidebar.slider,st.sidebar.selectbox dll

def user_input_features():
    starRating = st.sidebar.slider('Star Rating',0,5,3) #(label,minvalues,maxvalues,initial values)
    builtYear = st.sidebar.slider('Built Year',1900,2023,1960)
    size = st.sidebar.slider("Room Size (m2)",2.0,100.0,50.0,0.1,format='%0.1f')#(label,minvalues,maxvalues,initial values,jarak increment,format spesifik)
    occupancy = st.sidebar.slider('Occupancy',1,5,3)
    childAge = st.sidebar.slider('Child Age',0,18,9)
    childOccupancy = st.sidebar.slider('Child Occupancy',0,5,2)
    breakfast = st.sidebar.checkbox('Breakfast Include')
    wifi = st.sidebar.checkbox("Wifi Include")
    refund = st.sidebar.checkbox('Refund')
    livingRoom = st.sidebar.checkbox('Living Room')
    hotelFacilitie = st.sidebar.multiselect('Hotel Facilities',(hotelFacilities))
    roomFacilitie = st.sidebar.multiselect('Room Faciclities', (roomFacilities))
    pointInterest = st.sidebar.multiselect('Point of Interest',(nearestPoint))

    # handle checkbox
    breakfast = 1 if breakfast else 0
    wifi = 1 if wifi else 0
    refund = 1 if refund else 0
    livingRoom = 1 if livingRoom else 0

    # handle MultiSelect
    hotelFacilitie = ','.join(hotelFacilitie)
    roomFacilitie = ','.join(roomFacilitie)
    pointInterest = ','.join(pointInterest)

    data = {'starRating': starRating,
            'builtYear': builtYear,
            'size': size,
            'baseOccupancy': occupancy,
            'maxChildAge': childAge,
            'maxChildOccupancy': childOccupancy,
            'isBreakfastIncluded': breakfast,
            'isWifiIncluded': wifi,
            'isRefundable': refund,
            'hasLivingRoom': livingRoom,
            'hotelFacilities': hotelFacilitie,
            'roomFacilities': roomFacilitie,
            'nearestPoint': pointInterest
            }
    
    features = pd.DataFrame(data,index=[0])
    return features



df = user_input_features()
st.header("User Input Features")
st.write(df)
# handling input user
# buat fungsi untuk membuat dataframe dengan nilai 0 dan 1

def create_df(dfOri,df_name,df,prefix):
    value = prefix+dfOri[df_name][0]
    for i in range(0,len(df.columns)):
        column_name = df.columns[i]
        if column_name in value:
            df.loc[0,column_name] = 1
        else:
            df.loc[0,column_name] = 0

    return df

# buat dataframe kosong untuk hotelfacilities,roomFacilitis,nearestPoint 
# dengan nama kolom dari hotelFacilities, roomFacilities, nearestPoint
roomFacilities_df = pd.DataFrame(columns=roomFacilities)
hotelFacilities_df = pd.DataFrame(columns=hotelFacilities)
nearestPoint_df = pd.DataFrame(columns=nearestPoint)

create_df(df,'roomFacilities',roomFacilities_df, 'Room_')
create_df(df,'hotelFacilities',hotelFacilities_df,'Hotel_')
create_df(df,'nearestPoint',nearestPoint_df,'Point_')


# menghapus kolom hotelFacilities,room facilities,nearestpoint
# lalu gantikan dengan df yang values roomfasilities, hotelfacilities dan nearest point sudah diganti 0 dan 1
# lalu gabungkan 
df = df.drop(['hotelFacilities','roomFacilities','nearestPoint'],axis = 1)
df = pd.concat([df,hotelFacilities_df,roomFacilities_df,nearestPoint_df],axis=1)

# change all column data type to unit8 kecuali kolom pertama
df= df.astype({col: 'float64' for col in df.columns[:2]})
df= df.astype({col: 'uint8' for col in df.columns[2:]})

# mengecek dataframe
# apakah dataframe sesuai yang digunakan saat traning
# check df columgn order with model column order using colOri , 
# jika tidak sama print kolom yang salah
# colOri merupakan kolom pada data training yang di export menggunakan pickle

colOri = colOri[1:]
# if df.columns.tolist() == colOri.all():
#     st.info("Column order is correct.")
# else:
#     mismatched_columns = [(idx, df_col, model_col) for idx, (df_col, model_col) in enumerate(zip(df.columns.tolist(), colOri)) if df_col != model_col]

#     if len(mismatched_columns) > 0:
#         st.warning("The order of the columns is not the same as the model. Mismatched columns:")
#         for idx, df_col, model_col in mismatched_columns:
#             st.write(f"At index {idx}: DataFrame column '{df_col}' - Model column '{model_col}'")


if df.columns.tolist() == colOri.tolist():
    st.info("Column order is correct.")
else:
    st.warning("Column order is incorrect. Please check the columns.")


st.write('press button below to predict : ')
model = st.selectbox('Select Model',('XGBoost','Random Forest','SVR'))

if model == 'XGBoost' and st.button('Predict'):
    # create progres bar widget with initial progress is 0%
    bar = st.progress(0)
    # create an empty container or space
    status_text = st.empty()
    for i in range(1,101):
        # create a text to showing a percentage process
        status_text.text("%i%% complete" %i)
        # give bar progress values
        bar.progress(i)
        # give bar progress time to execute the values
        time.sleep(0.01)

    #formatting the prediction
    prediction = xgbModel.predict(df)
    # "{:": This is the start of the format specifier.
    # ",": This specifies that a comma should be used as a thousands separator. In many countries, a comma is used to separate thousands in large numbers, making them easier to read.
    # "2f": This specifies how to format the floating-point number. In this case, it's using 2 decimal places (i.e., it will show two digits after the decimal point).
    formatString = "Rp{:,.2f}"
    #  change the format of prediction variable to float
    prediction = float(prediction[0])
    formatted_prediction = formatString.format(prediction)
    time.sleep(0.08)

    # print the prediction
    st.subheader('Prediction')
    st.metric('Price (IDR)',formatted_prediction)


elif model == 'Random Forest' and st.button('Predict'):
    bar = st.progress(0)
    status_text = st.empty()
    for i in range(1, 101):
        status_text.text("%i%% Complete" % i)
        bar.progress(i)
        time.sleep(0.01)

    # Formatting the prediction
    prediction = rfModel.predict(df)
    formaString = "Rp{:,.2f}"
    prediction = float(prediction[0])
    formatted_prediction = formaString.format(prediction)
    time.sleep(0.08)

    # print the prediction
    st.subheader('Prediction')
    st.metric('Price (IDR)', formatted_prediction)

    # empty the progress bar and status text
    time.sleep(0.08)
    bar.empty()
    status_text.empty()

elif model == 'SVR' and st.button('Predict'):
    bar = st.progress(0)
    status_text = st.empty()
    for i in range(1, 101):
        status_text.text("%i%% Complete" % i)
        bar.progress(i)
        time.sleep(0.01)

    # Formatting the prediction
    prediction = svrModel.predict(df)
    
    formaString = "Rp{:,.2f}"
    prediction = float(prediction[0])
    formatted_prediction = formaString.format(prediction)
    # prediction = rfModel.predict(df)
    time.sleep(0.08)

    # print the prediction
    st.subheader('Prediction')
    st.metric('Price (IDR)', formatted_prediction)

    # empty the progress bar and status text
    time.sleep(0.08)
    bar.empty()
    status_text.empty()
        