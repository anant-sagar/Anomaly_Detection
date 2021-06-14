import os
from numpy.core import numeric
from numpy.lib.twodim_base import diagflat
import streamlit as st
from detection import * 
from db import CsvData
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn import svm


st.image("aa.png")


def opendb():
    engine = create_engine('sqlite:///database.sqlite3') # connect
    Session =  sessionmaker(bind=engine)
    return Session()

def save_file(file,path):
    try:
        db = opendb()
        ext = file.type.split('/')[1] # second piece
        csvfile = CsvData(filename=file.name,extension=ext,filepath=path)
        db.add(csvfile)
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.write("database error:",e)
        return False

def delete_file(file):
    try:
        db = opendb()
        db.query(CsvData).filter(CsvData.id == file.id).delete()
        if os.path.exists(file.filepath):
            os.unlink(file.filepath)
        db.commit()
        db.close()
        st.info("csv file deleted")
    except Exception as e:
        st.error("csv file could not deleted")
        st.error(e)



#choice = st.sidebar.selectbox("select option",['about project','add new dataset','view datasets','detect anomaly','delete datasets'])

if st.checkbox('About Project'):
    st.sidebar.success("""Before doing any data analysis, the need to find out any outliers in a dataset arises. These outliers are known as anomalies. 

The goals of anomaly detection and outlines the approaches used to solve specific use cases for anomaly detection and condition monitoring.""")
    st.markdown("""The main goal of Anomaly Detection analysis is to identify the observations that do not adhere to general patterns considered as normal behavior.    """)
    st.markdown("""There are two directions in data analysis that search for anomalies: outlier detection and novelty detection. So, the outlier is the observation that differs from other data points in the train dataset. The novelty data point also differs from other observations in the dataset, but unlike outliers, novelty points appear in the test dataset and usually absent in the train dataset.""")
    st.markdown("""The most common reason for the outliers are;\n

data errors (measurement inaccuracies, rounding, incorrect writing, etc.);\n
noise data points;\n
hidden patterns in the dataset (fraud or attack requests).\n""")
    st.markdown("""So outlier processing depends on the nature of the data and the domain. Noise data points should be filtered (noise removal); data errors should be corrected. Some applications focus on anomaly selection, and we consider some applications further.  """)

if st.checkbox('Add New Dataset'):
    file = st.file_uploader("select a Csv Dataset",type=['csv'])
    if file:
        path = os.path.join('datasets',file.name)
        with open(path,'wb') as f:
            f.write(file.getbuffer())
            status = save_file(file,path)
            if status:
                st.sidebar.success("file uploaded")
                st.write(pd.read_csv(path).head(),use_column_width=True)
            else:
                st.sidebar.error('upload failed')

if st.checkbox('View Datasets'):
    db = opendb()
    results = db.query(CsvData).all()
    db.close()
    csv = st.sidebar.radio('select dataset file',results)
    if csv and os.path.exists(csv.filepath):
        st.sidebar.info("selected a csv dataset")
        st.write(pd.read_csv(csv.filepath), use_column_width=True)
        

if st.checkbox('Delete Datasets'):
    db = opendb()
    # results = db.query(Image).filter(Image.uploader == 'admin') if u want to use where query
    results = db.query(CsvData).all()
    db.close()
    csv = st.sidebar.radio('select dataset file to remove',results)
    if csv:
        st.error("csv file to be deleted")
        if os.path.exists(csv.filepath):
            st.write(pd.read_csv(csv.filepath).head(), use_column_width=True)
        if st.sidebar.button("delete"): 
            delete_file(csv)

if st.checkbox('Detect Aomaly'):
    db = opendb()
    results = db.query(CsvData).all()
    db.close()
    csv = st.sidebar.selectbox('Select dataset file to remove',results)
    if os.path.exists(csv.filepath):
        df = pd.read_csv(csv.filepath)
        numeric_cols = []
        for colname in df:
            if 'int' in str(df[colname].dtypes) or 'float' in str(df[colname].dtypes):
                numeric_cols.append(colname)
        df = df[numeric_cols]
        cols = df.columns.tolist()
        sel_cols = st.sidebar.multiselect("select any two columns",cols)
        algo = st.sidebar.selectbox('Select an Algorithm',['rbf','linear','poly'])
        gamma = st.sidebar.selectbox("Gamma value",['scale',1,.1,.01,.001])
        if len(sel_cols) == 2:
            try:
                data = df[[sel_cols[0],sel_cols[1]]]
                clf = svm.OneClassSVM(nu=0.05, kernel=algo, gamma=gamma)
                clf.fit(data)
                pred = clf.predict(data)
                normal = data[pred == 1]
                abnormal = data[pred == -1]
                fig,ax = plt.subplots(figsize=(10,7))
                normal.plot(kind='scatter',x=sel_cols[0],y=sel_cols[1],ax=ax,color='blue',alpha=.5,label='normal data')
                abnormal.plot(kind='scatter',x=sel_cols[0],y=sel_cols[1],ax=ax,color='red',s=100,label='anomaly/outlier')
                plt.legend()
                plt.xlabel(sel_cols[0])
                plt.ylabel(sel_cols[1])
                st.pyplot(fig,)
            except Exception as e:
                st.write("Please select other column",e)

        else:
            st.warning('Select two column to detect anomaly with visualization')
    else:
        st.error('The file was not found, reupload it')