#=============================================================================
# Import necessary packages
import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, CARTODBPOSITRON_RETINA
from bokeh.models import HoverTool, ColumnDataSource, ColorBar, CustomJS, DataTable, TableColumn, HTMLTemplateFormatter
from bokeh.palettes import Plasma10, Spectral11
from bokeh.transform import linear_cmap
from streamlit_bokeh_events import streamlit_bokeh_events
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from streamlit_option_menu import option_menu

#=============================================================================
# --- Initializing SessionState ---
if "load_state" not in st.session_state:
     st.session_state.load_state = False

@st.cache
def load_data(what_data):
    if what_data == 'map_data' :
        data = pd.read_csv('assets/geom.csv', index_col='PID')
    elif what_data == 'side_data' :
        data = pd.read_csv('assets/lala.csv', index_col=0)
    elif what_data == 'address_data' :
        data = pd.read_csv('assets/addID.csv', index_col='PID')
    elif what_data == 'model_data' :
        data = pd.read_csv('assets/pkl_base.csv', index_col='PID')
    return data


# Loading Features used in ElasticNet model
with open('assets/features_used.txt') as f:
    features_used = f.readlines()
f.close()
# # Loading House Points
# with open('assets/house_points.txt') as f:
#     house_points = f.readlines()
# f.close()
# # Loading Map Outline Coordinates
# with open('assets/map_outline.txt') as f:
#     map_outline = f.readlines()
# f.close()

## Sidebar Data
avg = load_data('side_data')
## Address Data
addID = load_data('address_data')
## Model DATA
FinalData = load_data('model_data')
## MAP DATA
merged = load_data('map_data')

# Load the features used into a list
features = ['GrLivArea', 'SalePrice', 'LotFrontage', 'LotArea', 'LotShape', 'OverallQual', 'OverallCond', 'MasVnrArea', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'TotalBsmtSF', 'HeatingQC', 'LowQualFinSF', 'KitchenQual', 'TotRmsAbvGrd', 'FireplaceQu', 'GarageFinish', 'GarageArea', 'Fence', 'YrSold', 'RemodelBool', 'Age', 'Totalbathr', 'FinBsmt_Perc', 'Outside_Spaces', 'Condition_Norm', 'Condition_Feedr_Artery', 'Condition_PosAN', 'Condition_RRewns', 'MSZoning__RH', 'MSZoning__RM', 'Alley__1', 'LandContour__Bnk', 'LandContour__HLS', 'LandContour__Low', 'LotConfig__Corner', 'LotConfig__CulDSac', 'LotConfig__FR2', 'LotConfig__FR3', 'Neighborhood__Blmngtn', 'Neighborhood__Blueste', 'Neighborhood__BrDale', 'Neighborhood__BrkSide', 'Neighborhood__ClearCr', 'Neighborhood__CollgCr', 'Neighborhood__Crawfor', 'Neighborhood__Edwards', 'Neighborhood__Gilbert', 'Neighborhood__Greens', 'Neighborhood__GrnHill', 'Neighborhood__IDOTRR', 'Neighborhood__Landmrk', 'Neighborhood__MeadowV', 'Neighborhood__Mitchel', 'Neighborhood__NPkVill', 'Neighborhood__NWAmes', 'Neighborhood__NoRidge', 'Neighborhood__NridgHt', 'Neighborhood__OldTown', 'Neighborhood__SWISU', 'Neighborhood__Sawyer', 'Neighborhood__SawyerW', 'Neighborhood__Somerst', 'Neighborhood__StoneBr', 'Neighborhood__Timber', 'Neighborhood__Veenker', 'HouseStyle__1.5Fin', 'HouseStyle__1.5Unf', 'HouseStyle__2.5Fin', 'HouseStyle__2.5Unf', 'HouseStyle__2Story', 'HouseStyle__SFoyer', 'HouseStyle__SLvl', 'RoofStyle__Flat', 'RoofStyle__Gambrel', 'RoofStyle__Hip', 'RoofStyle__Mansard', 'RoofStyle__Shed', 'MasVnrType__BrkCmn', 'MasVnrType__BrkFace', 'MasVnrType__Stone', 'CentralAir__0', 'Electrical__0', 'PavedDrive__0', 'SaleType__Contract', 'SaleType__Other', 'PoolArea__1', 'Foundation__BrkTil', 'Foundation__PConc', 'Foundation__Slab', 'Foundation__Stone', 'Foundation__Wood', 'Season__Autumn', 'Season__Spring', 'Season__Winter', 'SaleCondition__Other', 'NeighborhoodSafety__1', 'NeighborhoodSafety__2', 'NeighborhoodSafety__3', 'NeighborhoodSafety__4', 'NeighborhoodSafety__5', 'NeighborhoodSafety__7', 'NeighborhoodSD__5']

#=============================================================================
# Navigation
st.sidebar.image('assets/AppLogo.png', use_column_width=True) 

with st.sidebar.container():
     st.sidebar.title(' ABOUT: \n ***Founded in 2022, MSJL Consulting Co. helps home buyers estimate home prices and homeowners estimate the price change after remodelling***')

page = st.sidebar.radio("Menu", ["Map of Ames", "Price Predictions", "Remodelling"]) 

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

neib_fullname = {'Blmngtn':'Bloomington Heights',
       'Blueste':'Bluestem',
       'BrDale':'Briardale',
       'BrkSide':'Brookside',
       'ClearCr':'Clear Creek',
       'CollgCr':'College Creek',
       'Crawfor':'Crawford',
       'Edwards':'Edwards',
       'Gilbert':'Gilbert',
       'IDOTRR':'Iowa DOT and Rail Road',
       'MeadowV':'Meadow Village',
       'Mitchel':'Mitchell',
       'NAmes':'North Ames',
       'NoRidge':'Northridge',
       'NPkVill':'Northpark Villa',
       'NridgHt':'Northridge Heights',
       'NWAmes':'Northwest Ames',
       'OldTown':'Old Town',
       'SWISU':'South West of ISU',
       'Sawyer':'Sawyer',
       'SawyerW':'Sawyer West',
       'Somerst':'Somerset',
       'StoneBr':'Stone Brook',
       'Timber':'Timberland',
       'Veenker':'Veenker',
       'Greens':'Greensboro',
       'GrnHill':'Greens Hills',
       'Landmrk':'Landmark Villas'}

# ========Modeling Functions===================================================
def num_format(num):
    # converts any int/float to human readable string with thousandth commas
    new_num = ''
    for idx, c in enumerate(str(np.int64(num))[::-1]):
        if (idx+1)%4 == 0:
            new_num += ','
        new_num += c
    return new_num[::-1]

def pkl_dum_encode(base_data, code, feat):
    # Encodes the feature selected with '1', all other dummy columns are set to '0'
    reg_text = '^'+feat
    target = feat+code
    feat_cols = list(base_data.filter(regex=reg_text).columns)
    base_data.loc[0,feat_cols] = 0
    if target in feat_cols:
        feat_cols.remove(target)
        base_data.loc[0,target] = 1
    return base_data

basehouse_PIN = 535454150

## model function
def elasticnet(df, sdf):
    x = df.drop(['SalePrice'],axis=1)
    y = df['SalePrice']
    y=np.log(y)

    #to convert whatever strings your data might contain to numeric values. 
    #If they're incompatible with conversion, they'll be reduced to NaNs.
    x = x.apply(pd.to_numeric, errors='coerce')
    y = y.apply(pd.to_numeric, errors='coerce')
    x.fillna(0, inplace=True)
    y.fillna(0, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 41)
    model = ElasticNet(alpha=0.000041775510204081, l1_ratio=0.9591836734693877, normalize=True)
    model.fit(x_train, y_train)
    
    j = sdf.drop(['SalePrice'],axis=1)

    return model.predict(j)

## predict function
def elasticnet2(df, sdf):
    x = df.drop(['SalePrice'],axis=1)
    y = df['SalePrice']
    y=np.log(y)

    #to convert whatever strings your data might contain to numeric values. 
    #If they're incompatible with conversion, they'll be reduced to NaNs.
    x = x.apply(pd.to_numeric, errors='coerce')
    y = y.apply(pd.to_numeric, errors='coerce')
    x.fillna(0, inplace=True)
    y.fillna(0, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 41)
    model = ElasticNet(alpha=0.000041775510204081, l1_ratio=0.9591836734693877, normalize=True)
    model.fit(x_train, y_train)
    
    j = sdf.drop(['SalePrice'])

    return model.predict(j.values.reshape(1, -1))

#=============================================================================
# Map of Ames
if page == "Map of Ames":
    with st.container():
        st.title('Map of Ames')
        st.session_state.load_state = False
        fig = go.Figure(data=[go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Blmngtn']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Blmngtn']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Bloomington Heights', hovertemplate =  "<b>Neighborhood: </b> Bloomington Heights <br>" +"<b>Min Price : </b> 159895 <br>" + "<b>Max Price : </b> 246990 <br>"+"<b>1Fam Avg Price : </b> 177689 <br>" + "<b>TwnhsE Avg Price : </b> 199019 <br>") , go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='BrDale']['Longitude'], lat=merged.loc[merged['Neighborhood']=='BrDale']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Briardale', hovertemplate =  "<b>Neighborhood: </b> Briardale <br>" +"<b>Min Price : </b> 83000 <br>" + "<b>Max Price : </b> 125500 <br>"+"<b>Twnhs Avg Price : </b> 104467 <br>" + "<b>TwnhsE Avg Price : </b> 112333 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='ClearCr']['Longitude'], lat=merged.loc[merged['Neighborhood']=='ClearCr']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Clear Creek', hovertemplate =  "<b>Neighborhood: </b> Clear Creek <br>" +"<b>Min Price : </b> 107500 <br>" + "<b>Max Price : </b> 328000 <br>"+"<b>1Fam Avg Price : </b> 215662 <br>" + "<b>Twnhs Avg Price : </b> 148400 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Gilbert']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Gilbert']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Gilbert', hovertemplate =  "<b>Neighborhood: </b> Gilbert <br>" +"<b>Min Price : </b> 115000 <br>" + "<b>Max Price : </b> 377500 <br>"+"<b>1Fam Avg Price : </b> 189999 <br>" + "<b>2fmCon Avg Price : </b> 150000 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='NAmes']['Longitude'], lat=merged.loc[merged['Neighborhood']=='NAmes']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'North Ames', hovertemplate =  "<b>Neighborhood: </b> North Ames <br>" +"<b>Min Price : </b> 68000 <br>" + "<b>Max Price : </b> 345000 <br>"+"<b>1Fam Avg Price : </b> 146314 <br>" + "<b>2fmCon Avg Price : </b> 132000 <br>" +"<b>Duplex Avg Price : </b> 127547 <br>" + "<b>TwnhsE Avg Price : </b> 167800 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='NPkVill']['Longitude'], lat=merged.loc[merged['Neighborhood']=='NPkVill']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Northpark Villa', hovertemplate =  "<b>Neighborhood: </b> Northpark Villa <br>" +"<b>Min Price : </b> 120000 <br>" + "<b>Max Price : </b> 155000 <br>"+"<b>Twnhs Avg Price : </b> 189999 <br>" + "<b>TwnhsE Avg Price : </b> 150000 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='NWAmes']['Longitude'], lat=merged.loc[merged['Neighborhood']=='NWAmes']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Northwest Ames', hovertemplate =  "<b>Neighborhood: </b> Northwest Ames <br>" +"<b>Min Price : </b> 82500 <br>" + "<b>Max Price : </b> 306000 <br>"+"<b>1Fam Avg Price : </b> 192074 <br>" + "<b>2fmCon Avg Price : </b> 146500 <br>" +"<b>Duplex Avg Price : </b> 137468 <br>" ), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='StoneBr']['Longitude'], lat=merged.loc[merged['Neighborhood']=='StoneBr']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Stone Brook', hovertemplate =  "<b>Neighborhood: </b> Stone Brook <br>" +"<b>Min Price : </b> 150000 <br>" + "<b>Max Price : </b> 591587 <br>"+"<b>1Fam Avg Price : </b> 380730 <br>" + "<b>TwnhsE Avg Price : </b> 233313 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Greens']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Greens']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Greensboro', hovertemplate =  "<b>Neighborhood: </b> Greensboro <br>" +"<b>Min Price : </b> 155000 <br>" + "<b>Max Price : </b> 214000 <br>"+"<b>Twnhs Avg Price : </b> 194500 <br>" + "<b>TwnhsE Avg Price : </b> 192562 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='NridgHt']['Longitude'], lat=merged.loc[merged['Neighborhood']=='NridgHt']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Northridge Heights', hovertemplate =  "<b>Neighborhood: </b> Northridge Heights <br>" +"<b>Min Price : </b> 154000 <br>" + "<b>Max Price : </b> 615000 <br>"+"<b>1Fam Avg Price : </b> 354578 <br>" + "<b>Twnhs Avg Price : </b> 191541 <br>" + "<b>TwnhsE Avg Price : </b> 234802 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='NoRidge']['Longitude'], lat=merged.loc[merged['Neighborhood']=='NoRidge']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Northridge', hovertemplate =  "<b>Neighborhood: </b> Northridge <br>" +"<b>Min Price : </b> 190000 <br>" + "<b>Max Price : </b> 755000 <br>"+"<b>1Fam Avg Price : </b> 326114 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Sawyer']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Sawyer']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Sawyer', hovertemplate =  "<b>Neighborhood: </b> Sawyer <br>" +"<b>Min Price : </b> 62383 <br>" + "<b>Max Price : </b> 219000 <br>"+"<b>1Fam Avg Price : </b> 137760 <br>" + "<b>2fmCon Avg Price : </b> 124100 <br>" + "<b>Duplex Avg Price : </b> 139340 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='SawyerW']['Longitude'], lat=merged.loc[merged['Neighborhood']=='SawyerW']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Sawyer West', hovertemplate =  "<b>Neighborhood: </b> Sawyer West <br>" +"<b>Min Price : </b> 67500 <br>" + "<b>Max Price : </b> 320000 <br>"+"<b>1Fam Avg Price : </b> 191220 <br>" + "<b>Duplex Avg Price : </b> 211421 <br>" + "<b>TwnhsE Avg Price : </b> 146327 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Somerst']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Somerst']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Somerset', hovertemplate =  "<b>Neighborhood: </b> Somerset <br>" +"<b>Min Price : </b> 139000 <br>" + "<b>Max Price : </b> 468000 <br>"+"<b>1Fam Avg Price : </b> 251257 <br>" + "<b>Twnhs Avg Price : </b> 174018 <br>" + "<b>TwnhsE Avg Price : </b> 201277 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Veenker']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Veenker']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Veenker', hovertemplate =  "<b>Neighborhood: </b> Veenker <br>" +"<b>Min Price : </b> 139000 <br>" + "<b>Max Price : </b> 385000 <br>"+"<b>1Fam Avg Price : </b> 246797 <br>" + "<b>TwnhsE Avg Price : </b> 267340 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='BrkSide']['Longitude'], lat=merged.loc[merged['Neighborhood']=='BrkSide']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#F5F2A1'), name = 'Brookside', hovertemplate =  "<b>Neighborhood: </b> Brookside <br>" +"<b>Min Price : </b> 39300 <br>" + "<b>Max Price : </b> 223500 <br>"+"<b>1Fam Avg Price : </b> 126079 <br>" + "<b>2fmCon Avg Price : </b> 123500 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='IDOTRR']['Longitude'], lat=merged.loc[merged['Neighborhood']=='IDOTRR']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#F5F2A1'), name = 'Iowa DOT and Rail Road', hovertemplate =  "<b>Neighborhood: </b> Iowa DOT and Rail Road <br>" +"<b>Min Price : </b> 34900 <br>" + "<b>Max Price : </b> 212300 <br>"+"<b>1Fam Avg Price : </b> 109321 <br>" + "<b>2fmCon Avg Price : </b> 95996 <br>" +"<b>Duplex Avg Price : </b> 110000 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Landmrk']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Landmrk']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#F5F2A1'), name = 'Landmark Villas', hovertemplate =  "<b>Neighborhood: </b> Landmark Villas <br>" +"<b>Min Price : </b> 137000 <br>" + "<b>Max Price : </b> 137000 <br>"+"<b>Twnhs Avg Price : </b> 137000 <br>" ), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='OldTown']['Longitude'], lat=merged.loc[merged['Neighborhood']=='OldTown']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#F5F2A1'), name = 'Old Town', hovertemplate =  "<b>Neighborhood: </b> Old Town <br>" +"<b>Min Price : </b> 12789 <br>" + "<b>Max Price : </b> 475000 <br>"+"<b>1Fam Avg Price : </b> 127045 <br>" + "<b>2fmCon Avg Price : </b> 127368 <br>" +"<b>Duplex Avg Price : </b> 121100 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='GrnHill']['Longitude'], lat=merged.loc[merged['Neighborhood']=='GrnHill']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'Greens Hills', hovertemplate =  "<b>Neighborhood: </b> Greens Hills <br>" +"<b>Min Price : </b> 230000 <br>" + "<b>Max Price : </b> 330000 <br>"+"<b>TwnhsE Avg Price : </b> 280000 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Blueste']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Blueste']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'Bluestem', 
hovertemplate = "<b>Neighborhood: </b> Bluestem <br>" +"<b>Min Price : </b> 115000 <br>" + "<b>Max Price : </b> 200000 <br>"+"<b>Twnhs Avg Price : </b> 1125480 <br>" + "<b>TwnhsE Avg Price : </b> 161700 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='CollgCr']['Longitude'], lat=merged.loc[merged['Neighborhood']=='CollgCr']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'College Creek', hovertemplate =  "<b>Neighborhood: </b> College Creek <br>" +"<b>Min Price : </b> 110000 <br>" + "<b>Max Price : </b> 475000 <br>"+"<b>1Fam Avg Price : </b> 201432 <br>" + "<b>Duplex Avg Price : </b> 185000 <br>"+"<b>TwnhsE Avg Price : </b> 142803 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Crawfor']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Crawfor']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'Crawford', hovertemplate =  "<b>Neighborhood: </b> Crawford <br>" +"<b>Min Price : </b> 90350 <br>" + "<b>Max Price : </b> 392500 <br>"+"<b>1Fam Avg Price : </b> 198512 <br>" + "<b>2fmCon Avg Price : </b> 148500 <br>"+"<b>Duplex Avg Price : </b> 177500 <br>" + "<b>TwnhsE Avg Price : </b> 283083 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Edwards']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Edwards']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'Edwards', hovertemplate =  "<b>Neighborhood: </b> Edwards <br>" +"<b>Min Price : </b> 35000 <br>" + "<b>Max Price : </b> 415000 <br>"+"<b>1Fam Avg Price : </b> 134800 <br>" + "<b>2fmCon Avg Price : </b> 106380 <br>"+"<b>Duplex Avg Price : </b> 121108 <br>" + "<b>Twnhs Avg Price : </b> 132875 <br>" +"<b>TwnhsE Avg Price : </b> 140062 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='SWISU']['Longitude'], lat=merged.loc[merged['Neighborhood']=='SWISU']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'South & West of ISU', hovertemplate =  "<b>Neighborhood: </b> South & West of ISU <br>" +"<b>Min Price : </b> 60000 <br>" + "<b>Max Price : </b> 197000 <br>"+"<b>1Fam Avg Price : </b> 133739 <br>" + "<b>2fmCon Avg Price : </b> 123860 <br>"+"<b>Duplex Avg Price : </b> 155000 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Timber']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Timber']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'Timberland', hovertemplate =  "<b>Neighborhood: </b> Timberland <br>" +"<b>Min Price : </b> 150000 <br>" + "<b>Max Price : </b> 425000 <br>"+"<b>1Fam Avg Price : </b> 243189 <br>" + "<b>2fmCon Avg Price : </b> 228950 <br>"+"<b>TwnhsE Avg Price : </b> 242750 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='MeadowV']['Longitude'], lat=merged.loc[merged['Neighborhood']=='MeadowV']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('pink'), name = 'Meadow Village', hovertemplate =  "<b>Neighborhood: </b> Meadow Village <br>" +"<b>Min Price : </b> 73000 <br>" + "<b>Max Price : </b> 151400 <br>"+"<b>Twnhs Avg Price : </b> 89299 <br>" + "<b>TwnhsE Avg Price : </b> 104373 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Mitchel']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Mitchel']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('pink'), name = 'Mitchell', hovertemplate =  "<b>Neighborhood: </b> Mitchell <br>" +"<b>Min Price : </b> 81500 <br>" + "<b>Max Price : </b> 300000 <br>"+"<b>1Fam Avg Price : </b> 164066 <br>" + "<b>2fmCon Avg Price : </b> 168000 <br>"+"<b>Duplex Avg Price : </b> 152940 <br>" + "<b>Twnhs Avg Price : </b> 164000 <br>" +"<b>TwnhsE Avg Price : </b> 156200 <br>")]).update_layout(plot_bgcolor='white',autosize=False, width = 1000, height = 800, legend=dict(x=1.0, y=1)).update_yaxes(showgrid=False).update_xaxes(showgrid=False)

        fig.update_layout(autosize=True, hovermode='closest',mapbox=dict( accesstoken="pk.eyJ1IjoibGF5YWxoYW1tYWQiLCJhIjoiY2wzbWd5ZWxjMDFqNDNmcWt5MzRzNHdlaCJ9.B04CAbFi5Llmk2B78EP6JQ", bearing=0,center=dict(lat=42.027269,lon= -93.611500), pitch=0,zoom=11.4,layers = [{'source': { "type":"GeometryCollection","geometries":[{"type":"MultiPolygon","coordinates": [[[[-93.6786634,42.0120788],[-93.6786591,42.0121036],[-93.6786609,42.0121814],[-93.6788265,42.0121812],[-93.6790274,42.0116318],[-93.6791176,42.0088862],[-93.688327,42.0116956],[-93.689336,42.0119745],[-93.6931857,42.0131309],[-93.6982674,42.0146493],[-93.6987155,42.0147872],[-93.6987173,42.014914],[-93.6987044,42.0153396],[-93.698705,42.0155209],[-93.6982904,42.0155213],[-93.695966,42.0155402],[-93.6938007,42.0155578],[-93.6932125,42.0155627],[-93.6932275,42.0181393],[-93.6932455,42.0212489],[-93.6932318,42.0229227],[-93.6932028,42.026452],[-93.6987325,42.0264162],[-93.6987274,42.0287827],[-93.6863508,42.0288983],[-93.68638,42.0282167],[-93.6861369,42.0282207],[-93.6861435,42.026511],[-93.6883586,42.0264918],[-93.6883466,42.0231315],[-93.6850045,42.023169],[-93.6850051,42.0233256],[-93.6846812,42.0233304],[-93.6846864,42.0234212],[-93.6848929,42.0284426],[-93.6860022,42.0284273],[-93.6860186,42.0300831],[-93.6883782,42.0300671],[-93.688367,42.0344929],[-93.689204,42.0344902],[-93.6937099,42.03453],[-93.695986,42.0345427],[-93.6959944,42.0355043],[-93.6970517,42.0355115],[-93.6970839,42.0389488],[-93.6960252,42.0388341],[-93.6960238,42.038923],[-93.6932727,42.0386564],[-93.6932724,42.0385943],[-93.6888904,42.0381866],[-93.68889,42.0380282],[-93.6838956,42.0376403],[-93.6838939,42.0378772],[-93.6810623,42.0375676],[-93.6808907,42.0379807],[-93.6787917,42.0379781],[-93.6787968,42.0372235],[-93.672288,42.0365268],[-93.6722978,42.0382344],[-93.6681735,42.0383083],[-93.6682758,42.0438392],[-93.6651378,42.0439077],[-93.665147,42.0471373],[-93.6590243,42.0471851],[-93.6592005,42.0525888],[-93.6592871,42.0562454],[-93.6555639,42.0562682],[-93.6558734,42.0563047],[-93.6560259,42.0563303],[-93.6561767,42.0563606],[-93.6564818,42.0564359],[-93.6567862,42.0565312],[-93.6569339,42.0565862],[-93.6573752,42.0567814],[-93.6576739,42.0569387],[-93.6578157,42.0570247],[-93.6580828,42.0572109],[-93.6582098,42.0573116],[-93.6583303,42.0574168],[-93.6584438,42.057526],[-93.6585503,42.0576392],[-93.6586494,42.0577561],[-93.6587463,42.0578857],[-93.6588357,42.0580183],[-93.6589172,42.0581537],[-93.6589909,42.0582916],[-93.6590564,42.0584317],[-93.6591137,42.0585716],[-93.6591623,42.0587133],[-93.6592022,42.0588566],[-93.6593035,42.0591272],[-93.6592325,42.0581754],[-93.659209,42.0569313],[-93.6602468,42.0569201],[-93.6608937,42.057763],[-93.6609222,42.0581576],[-93.6610335,42.0581564],[-93.6610296,42.0579355],[-93.6641295,42.0579102],[-93.6668399,42.0579186],[-93.6667062,42.0581438],[-93.6666479,42.0583333],[-93.6665923,42.0585145],[-93.6670973,42.0594701],[-93.6675365,42.0607608],[-93.6676923,42.0609293],[-93.6678374,42.0610862],[-93.6679998,42.0612618],[-93.6682481,42.0614395],[-93.6684574,42.061611],[-93.668691,42.0617181],[-93.6642084,42.0617164],[-93.6641869,42.0671091],[-93.6609464,42.067082],[-93.6609453,42.0659341],[-93.660232,42.0659282],[-93.6602292,42.0653233],[-93.6593515,42.0653158],[-93.6593436,42.0647053],[-93.6593421,42.0634544],[-93.6520439,42.0634925],[-93.6519328,42.063493],[-93.647781,42.0635147],[-93.6468391,42.0622502],[-93.6460037,42.0611572],[-93.64428,42.0586236],[-93.6433471,42.0573814],[-93.6429723,42.0573819],[-93.6456811,42.0612424],[-93.6461433,42.0618649],[-93.6466953,42.0626195],[-93.6473184,42.0635106],[-93.6400093,42.0635526],[-93.6378846,42.0635459],[-93.6378848,42.0636809],[-93.6374944,42.0636766],[-93.6374235,42.0671474],[-93.6399996,42.067163],[-93.6399857,42.0688392],[-93.6400012,42.0707136],[-93.6448653,42.0707148],[-93.644854,42.0671298],[-93.6496893,42.0671262],[-93.6497218,42.0761159],[-93.6448822,42.0760921],[-93.6448879,42.077915],[-93.6437047,42.0779055],[-93.6414291,42.0778652],[-93.6382357,42.077887],[-93.6349457,42.0779413],[-93.6331098,42.0779435],[-93.6309462,42.0779405],[-93.6297396,42.0779586],[-93.6269429,42.0779466],[-93.6269525,42.0765314],[-93.6270062,42.0764434],[-93.6275165,42.0762885],[-93.6274772,42.0761139],[-93.6274629,42.0759778],[-93.6273561,42.0757357],[-93.6268864,42.0754054],[-93.6254862,42.0755709],[-93.6255176,42.0738011],[-93.6255243,42.0736371],[-93.6236938,42.0735828],[-93.622636,42.0726856],[-93.6221556,42.0726229],[-93.62324,42.0681271],[-93.6233807,42.0669846],[-93.6235461,42.0634263],[-93.6237601,42.0634249],[-93.6240577,42.0615463],[-93.6235968,42.0615417],[-93.6235754,42.0621427],[-93.6235301,42.0624079],[-93.6230636,42.0623963],[-93.6230695,42.0620916],[-93.6224615,42.0620802],[-93.6224213,42.0618918],[-93.6223751,42.0615293],[-93.6203202,42.0615085],[-93.6202685,42.0615249],[-93.6202312,42.0616019],[-93.6201793,42.0616623],[-93.6201641,42.0617614],[-93.620141,42.0619761],[-93.6201033,42.0621577],[-93.6201098,42.0623615],[-93.620131,42.0625873],[-93.62016,42.0627196],[-93.6201814,42.0628959],[-93.6202469,42.0631053],[-93.620261,42.0632706],[-93.6202288,42.0633975],[-93.6157764,42.0634773],[-93.615717,42.0583343],[-93.6174111,42.0583791],[-93.6174831,42.0585974],[-93.6176598,42.0587136],[-93.6179471,42.0588632],[-93.6180651,42.0588911],[-93.6183008,42.0590075],[-93.6188096,42.0591413],[-93.6190085,42.0592355],[-93.6191854,42.0592967],[-93.6192678,42.0590216],[-93.618995,42.0589381],[-93.6188107,42.058888],[-93.6186117,42.0588158],[-93.6184349,42.0587216],[-93.6179121,42.0584281],[-93.6178681,42.0583453],[-93.6178177,42.0580533],[-93.6178115,42.0577779],[-93.6177681,42.0575685],[-93.6176582,42.0573864],[-93.6174153,42.057215],[-93.6171722,42.0570986],[-93.6169143,42.0570041],[-93.6166635,42.0569427],[-93.6163758,42.0569088],[-93.6160434,42.0569463],[-93.6157774,42.0570115],[-93.6155557,42.0570494],[-93.6154302,42.05706],[-93.6149358,42.0570199],[-93.6146557,42.0569144],[-93.6143537,42.0567647],[-93.6141478,42.0565823],[-93.6141045,42.0563564],[-93.6141274,42.0561913],[-93.6142096,42.0559547],[-93.6143436,42.0557073],[-93.6144326,42.055614],[-93.6145882,42.0554768],[-93.614781,42.0552682],[-93.6149513,42.0551586],[-93.6151071,42.0549883],[-93.6152483,42.0547685],[-93.6154419,42.0543836],[-93.6154727,42.0540808],[-93.6154299,42.0537503],[-93.6153348,42.0535572],[-93.6152911,42.0534194],[-93.6149759,42.0529007],[-93.6147186,42.0526631],[-93.6145713,42.0526021],[-93.6143574,42.0525628],[-93.6140844,42.0525234],[-93.6137818,42.0525225],[-93.6135087,42.0525106],[-93.61328,42.0524878],[-93.6130672,42.0523983],[-93.6130295,42.0523824],[-93.6127351,42.0521832],[-93.6125438,42.0520449],[-93.612309,42.0517082],[-93.6119788,42.0512666],[-93.611355,42.0504606],[-93.6106464,42.0504418],[-93.6104822,42.0498096],[-93.6099503,42.0494468],[-93.6033371,42.0492806],[-93.6033752,42.0458308],[-93.595979,42.0458064],[-93.5959434,42.045509],[-93.6008996,42.0455187],[-93.600843,42.0393042],[-93.5963053,42.0391164],[-93.5963214,42.0388301],[-93.5975165,42.0380951],[-93.597458,42.0372312],[-93.5959919,42.0351006],[-93.5959867,42.0342771],[-93.5870479,42.0341582],[-93.5870527,42.0344573],[-93.5891402,42.0344446],[-93.5890738,42.0360949],[-93.5864304,42.036097],[-93.5864206,42.0380303],[-93.5863693,42.0416673],[-93.5815337,42.041663],[-93.5815329,42.0480918],[-93.585102,42.0481311],[-93.5850919,42.0490629],[-93.5815293,42.0490295],[-93.5815022,42.0521992],[-93.5814951,42.0536524],[-93.5814853,42.0542131],[-93.5703278,42.054039],[-93.570395,42.0417379],[-93.5691689,42.0417491],[-93.5672906,42.0417662],[-93.5605404,42.0418403],[-93.5605373,42.0412437],[-93.5605535,42.038066],[-93.5605467,42.0359892],[-93.560574,42.0345766],[-93.5580928,42.0345859],[-93.5560215,42.034604],[-93.5560217,42.0341459],[-93.5550697,42.0341521],[-93.5550783,42.0346083],[-93.5544621,42.0346095],[-93.5544352,42.0334775],[-93.5543713,42.0307921],[-93.5533819,42.0307942],[-93.5520029,42.0307619],[-93.5491543,42.0306225],[-93.5477965,42.0305467],[-93.5458414,42.0304326],[-93.5458497,42.0305249],[-93.5436365,42.0303817],[-93.5436346,42.0303271],[-93.5430267,42.0302849],[-93.5422884,42.0302344],[-93.5410215,42.0301472],[-93.5410217,42.0301072],[-93.5401487,42.0300654],[-93.539124,42.0300096],[-93.5383963,42.0299661],[-93.5371477,42.029893],[-93.533085,42.0296954],[-93.5283365,42.0294531],[-93.5216085,42.0290697],[-93.5214227,42.0231075],[-93.5264642,42.0231047],[-93.5262611,42.0122168],[-93.5310604,42.0122111],[-93.5311232,42.0158764],[-93.5360675,42.0158636],[-93.5457422,42.0159061],[-93.560448,42.0158618],[-93.5715938,42.0158382],[-93.5715606,42.0131242],[-93.5715581,42.0123507],[-93.5715644,42.0116848],[-93.5717306,42.0110876],[-93.5720998,42.0104039],[-93.5726103,42.0097224],[-93.5737654,42.0087146],[-93.574699,42.0080942],[-93.5764024,42.0076343],[-93.5809828,42.0073314],[-93.5810098,42.0082698],[-93.5812104,42.0082674],[-93.5812129,42.008031],[-93.581202,42.007341],[-93.5812057,42.0073173],[-93.5860514,42.0070082],[-93.5860442,42.0063418],[-93.5815091,42.0066311],[-93.5813886,42.0065815],[-93.5813878,42.0059227],[-93.5813248,42.0056483],[-93.5811684,42.0056417],[-93.5811937,42.0066547],[-93.5796955,42.0067341],[-93.5796955,42.0051957],[-93.5811614,42.0051957],[-93.5811489,42.0040885],[-93.581166,42.0039753],[-93.5866544,42.0039564],[-93.5867967,42.0050947],[-93.5889435,42.0056275],[-93.5890482,42.0056356],[-93.5901986,42.0056763],[-93.5908582,42.0056577],[-93.5941598,42.0055836],[-93.5939533,42.0053231],[-93.5934058,42.0042108],[-93.5937805,42.0032877],[-93.5938054,42.0028702],[-93.5935366,42.0023922],[-93.59265,42.001624],[-93.5921399,42.0011449],[-93.5911917,41.9996619],[-93.5909722,41.9987664],[-93.59125,41.9979882],[-93.5913315,41.9974699],[-93.5956892,41.9975168],[-93.5956499,41.9860118],[-93.6101803,41.9860129],[-93.6112415,41.986013],[-93.6112402,41.9888501],[-93.612052,41.9888632],[-93.6120842,41.9893373],[-93.6174418,41.9893143],[-93.6174435,41.9902041],[-93.6189461,41.9902066],[-93.6199114,41.9876253],[-93.6198708,41.9793983],[-93.6236043,41.9793384],[-93.6244197,41.9749305],[-93.6294627,41.9761216],[-93.6294629,41.9775318],[-93.6294996,41.9801829],[-93.6278049,41.9801815],[-93.62805,41.9865541],[-93.6295953,41.986566],[-93.6393172,41.9865793],[-93.639333,41.98841],[-93.6393611,41.9900425],[-93.6393874,41.9931061],[-93.639402,41.993494],[-93.6402817,41.9934859],[-93.642332,41.9910803],[-93.644211,41.9910711],[-93.64429,41.997391],[-93.6460974,41.9973928],[-93.6460986,41.9925496],[-93.6491247,41.9925904],[-93.6491332,41.9906585],[-93.6491335,41.9865657],[-93.6538229,41.9866204],[-93.6539267,41.9938396],[-93.6590445,41.9938292],[-93.6615517,41.993865],[-93.664426,41.9938698],[-93.662169,41.9954831],[-93.6617692,41.9957901],[-93.6634786,41.995804],[-93.6635062,41.997068],[-93.6639478,41.9970674],[-93.6639495,41.9977482],[-93.6634784,41.9977488],[-93.6634832,41.9985398],[-93.6634891,41.9986455],[-93.6626349,41.9985664],[-93.6623483,41.9985233],[-93.6620324,41.9985125],[-93.6611362,41.9986617],[-93.6608802,41.9987272],[-93.6605603,41.9986723],[-93.659356,41.9985664],[-93.659358,41.9992356],[-93.6589665,41.9992287],[-93.6589668,41.9998686],[-93.658942,42.0002906],[-93.6577888,41.999403],[-93.6577581,41.9993016],[-93.6577357,41.999199],[-93.657716,41.9989916],[-93.6577344,41.9987879],[-93.6577573,41.9986889],[-93.6577076,41.9986082],[-93.6573188,41.9989946],[-93.6567052,41.999293],[-93.6562394,41.9992261],[-93.6561509,41.9994921],[-93.6556149,42.0004942],[-93.6550862,42.001284],[-93.654164,42.0025605],[-93.6538359,42.0029628],[-93.6533917,42.0032868],[-93.6527231,42.0036375],[-93.6522402,42.0039383],[-93.6517534,42.0041834],[-93.6512541,42.0043704],[-93.6507924,42.0045667],[-93.6504434,42.0048746],[-93.650132,42.005285],[-93.6500329,42.0056205],[-93.6507287,42.006037],[-93.6526957,42.0064462],[-93.6540701,42.0065189],[-93.6552943,42.0064986],[-93.6553318,42.0034568],[-93.6569872,42.0034545],[-93.6586567,42.0034452],[-93.6589041,42.0034405],[-93.658914,42.0040755],[-93.6618911,42.0040506],[-93.6618962,42.0024327],[-93.6608521,42.0024177],[-93.660849,42.0029211],[-93.6604778,42.0029423],[-93.6604847,42.0023924],[-93.6588974,42.002402],[-93.6588776,42.0011025],[-93.6687723,42.0011812],[-93.6688093,42.0048004],[-93.6688412,42.0072086],[-93.6711752,42.0072151],[-93.673508,42.0074139],[-93.6737999,42.0075724],[-93.6738065,42.0074123],[-93.6743163,42.0075077],[-93.6753219,42.0077608],[-93.6762593,42.008052],[-93.6774896,42.0084028],[-93.6786959,42.0087842],[-93.678705,42.0097219],[-93.6786634,42.0120788]]],[[[-93.6547535,41.9970318],[-93.6540883,41.9970794],[-93.6540411,42.0006597],[-93.6546708,42.0008694],[-93.6549002,42.0006008],[-93.6550134,42.0004412],[-93.6551678,42.0001921],[-93.6552599,41.9999263],[-93.6553224,41.9996929],[-93.6554011,41.999361],[-93.655552,41.9990484],[-93.6557607,41.9987731],[-93.6554267,41.9984821],[-93.6547197,41.9985027],[-93.6547643,41.9982266],[-93.6547535,41.9970318]]],[[[-93.5810989,42.035267],[-93.5810966,42.038365],[-93.5815408,42.0383591],[-93.5815548,42.035276],[-93.5810989,42.035267]]]]}]}, 'type': "line", 'below': "traces", 'color' :'#7392DA', 'opacity': 0.5}]))
        # Plot!
        st.plotly_chart(fig, use_container_width=True)

#=============================================================================
# Home Price Predictions
elif page == "Price Predictions":
    with st.container():
        #st.session_state.load_state = True
        #page = "Price Predictions"
        st.title('Home Price Predictions')
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        sec_select = st.selectbox('Select Sector',['North','North West','Downtown','South','South West', 'South East'])
        sec_mapper = {'Downtown':'Downtown','South':'South','South West':'South West','South East':'South East','North':'North','North West':'North West'}
        model_sec = sec_mapper[sec_select]
        model_neib = st.radio('Select Neighborhood',addID.loc[addID.Directions==model_sec]['Neighborhood'].unique())
        st.markdown(f"### {neib_fullname[model_neib]}")
        address_df = addID.loc[(addID['Directions']==model_sec) & (addID['Neighborhood']==model_neib)]
        col_main, col_empty1, col_b, col_empty2, col_e = st.columns([4,0.3,4,0.3,4]) #Set Columns
        col_main.markdown('##### House Details')
        col_b.markdown('##### Lot Details')
        col_e.markdown('##### House Features')

        # Initialize default variables based on average of neighborhood
        pkl_predhouse = FinalData.iloc[0].copy()
        for f in features:
            pkl_predhouse[f] = FinalData.loc[address_df.index][f].mean()
        
         #**********************
        # Total number of rooms
        tot_rooms = col_main.select_slider(
            'Select Number of Rooms',
            options = range(0,int(FinalData.loc[address_df.index]['TotRmsAbvGrd'].max())))
        pkl_predhouse['TotRmsAbvGrd'] = tot_rooms

        # Total number of bathrooms
        tot_bath = col_main.select_slider(
            'Select Number of Bathrooms',
            options = map(lambda x: x/10.0, range(0,int(10*(FinalData.loc[address_df.index]['Totalbathr'].max())),5)))
        pkl_predhouse['Totalbathr'] = tot_bath

        # Total number of garage car spaces
        tot_cars = col_main.select_slider(
            'Select Number of Garage Car Spaces',
            options = range(0,int(addID.loc[address_df.index]['GarageCars'].max())))
        car_dim = 320 # square footage of one car space on average
        tot_gar_area = car_dim*tot_cars
        pkl_predhouse['GarageArea'] = tot_gar_area

        # Total above-ground living area
        tot_liv_area = col_main.select_slider(
            'Select Ground Living Area (sf)',
            options = range(0,int(FinalData.loc[address_df.index]['GrLivArea'].max()),100))
        pkl_predhouse['GrLivArea'] = tot_liv_area

        # Total lot area
        tot_lot_area = col_b.select_slider(
            'Select Lot Area (sf)',
            options = range(0,int(FinalData.loc[address_df.index]['LotArea'].max()),100))
        pkl_predhouse['LotArea'] = tot_lot_area

        # Total lot frontage length
        tot_lot_length = col_b.select_slider(
            'Select Lot Frontage (ft)',
            options = range(0,int(FinalData.loc[address_df.index]['LotFrontage'].max()),10))
        pkl_predhouse['LotFrontage'] = tot_lot_length

        col_b.markdown('##### House Quality')

        # Overall quality of house
        overall_qual = col_b.select_slider(
            'Select Overall Quality of House (rating)',
            options = range(0,int(FinalData['OverallQual'].max())))
        pkl_predhouse['OverallQual'] = overall_qual

        # Age of house
        age = col_b.select_slider(
            'Select Approximate Age (years)',
            options = range(0,int(FinalData.loc[address_df.index]['Age'].max()),5))
        pkl_predhouse['Age'] = age

        # Presence of basement
        base_ques = col_e.radio('Basement?', ['Yes', 'No'])
        if base_ques == 'Yes':
            base_pres = FinalData.loc[address_df.index]['TotalBsmtSF'].mean()
        else:
            base_pres = 0
        pkl_predhouse['TotalBsmtSF'] = base_pres

        # Driveway paving
        pave_ques = col_e.radio('Driveway Paved?', ['Yes', 'No'])
        if pave_ques == 'Yes':
            pave_pres = 0
        else:
            pave_pres = 1
        pkl_predhouse['PavedDrive__0'] = pave_pres

        # Remodeled
        remo_ques = col_e.radio('Remodeled?', ['Yes', 'No'])
        if remo_ques == 'Yes':
            remo_pres = 0
        else:
            remo_pres = 1
        pkl_predhouse['RemodelBool'] = remo_pres
    
        # Base House MODEL PRICE
        pkl_predprice = np.floor(np.exp(elasticnet2(FinalData, pkl_predhouse)[0]))
        col_e.subheader(f'**${num_format(pkl_predprice)}**')
        col_e.caption('Baseline Price Prediction')
        
#=============================================================================
# Home Remodelling Estimates
elif page == "Remodelling" or st.session_state.load_state:
    with st.container():
        st.session_state.load_state = True
        page = "Remodelling"
        st.title('Remodelling')
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        sec_select = st.selectbox('Select Sector',['North','North West','Downtown','South','South West', 'South East'])
        sec_mapper = {'Downtown':'Downtown','South':'South','South West':'South West','South East':'South East','North':'North','North West':'North West'}
        model_sec = sec_mapper[sec_select]
        model_neib = st.radio('Select Neighborhood',addID.loc[addID.Directions==model_sec]['Neighborhood'].unique())
        st.markdown(f"### {neib_fullname[model_neib]}")
        address_df = addID.loc[(addID['Directions']==model_sec) & (addID['Neighborhood']==model_neib)]
        col_main, col_empty, col_b , col_e = st.columns([4,0.3,4,3]) #Set Columns
        col_main.markdown('##### House Selection')
        col_b.markdown('##### House Details')
        col_e.markdown('<p style="font-family:Courier; color:white; font-size: 20px;">c</p>',unsafe_allow_html=True)
        col_r, col_m, col_bpx, col_rpx = st.columns([3,2,2,2])
        col_r.markdown('##### Renovation')
        col_m.markdown('<p style="font-family:Courier; color:white; font-size: 20px;">c</p>',unsafe_allow_html=True)

         #**********************
        with col_main.container():

            address_df = addID.loc[(addID['Directions']==model_sec) &  (addID['Neighborhood']==model_neib)]
            source = ColumnDataSource(address_df)
            template = """
                 <div style="font-weight: 520; 
                     color: #5DB3CE"> 
                 <%= value %>
                 </div>
                 """
            formatter = HTMLTemplateFormatter(template=template)
            columns = [TableColumn(field="Address", title="Address", formatter=formatter)]

            #define events
            source.selected.js_on_change("indices",
                 CustomJS(args=dict(source=source),
                 code="""
                 document.dispatchEvent(
                 new CustomEvent("INDEX_SELECT", {detail: {data: source.selected.indices}})
                 )
                 """)
                 )

            mytable = DataTable(source=source, columns=columns, height=220)
            result = streamlit_bokeh_events(
                bokeh_plot=mytable, 
                events="INDEX_SELECT", 
                key="Address", 
                refresh_on_update=True, 
                debounce_time=0,
                override_height=220)

            if result:
                 if result.get("INDEX_SELECT"):
                    st.markdown(f'#### **{address_df.iloc[result.get("INDEX_SELECT")["data"],4].values[0]}**')
                    basehouse_PIN = address_df.index.values[result.get("INDEX_SELECT")["data"]][0]
            pkl_basehouse = FinalData.loc[[basehouse_PIN]]
            pkl_basehouse2 = addID.loc[[basehouse_PIN]]

        hstype_mapper = {'1Fam':'Single-family Detached House', '2FmCon':'Two-family Converted Houses',
                        'Duplx':'Duplex house', 
                        'TwnhsE':'Townhouse End Unit', 'Twnhs':'Townhouse Inside Unit'}
        col_main.caption(f"{hstype_mapper[pkl_basehouse2['BldgType'].values[0]]} in {neib_fullname[model_neib]}")

        Qual_mapper = {1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Excellent'}
        Qual_mapper2 = {0: 'No Exposure', 1: 'Minimum Exposure', 2: 'Average Exposure', 3: 'Good Exposure'}
        Qual_mapper3 = {1: 'Unfinished', 2: 'Rough Finished', 3: 'Finished'}
        pkl_renohouse = pkl_basehouse.copy()
        
        # HOUSE RENO Details
        col_b.markdown(f"Number of Rooms: **{num_format(pkl_basehouse['TotRmsAbvGrd'].values[0])}**")

        # Number of Bathrooms
        col_b.markdown(f"Bathrooms:  **{num_format(pkl_basehouse['Totalbathr'].values[0])}**")

        # Exterior Quality
        try:
            col_b.markdown(f"Exterior Material Quality: **{Qual_mapper[pkl_basehouse['ExterQual'].values[0]]}**")
            reno_Exterior = col_r.radio('Remodel Exterior Material',['No', 'Yes'])
            if reno_Exterior == 'Yes':
                pkl_renohouse['ExterQual'] = 5
        except:
            col_b.markdown(f"ExterQual: **None**")

        # Kitchen Quality
        try:
            col_b.markdown(f"Kitchen Quality: **{Qual_mapper[pkl_basehouse['KitchenQual'].values[0]]}**")
            reno_Kitchen = col_r.radio('Remodel Kitchen',['No', 'Yes'])
            if reno_Kitchen == 'Yes':
                pkl_renohouse['KitchenQual'] = 5
        except:
            col_b.markdown(f"Kitchen Quality: **None**")

        # Basement Condition
        try:
            if pkl_basehouse['TotalBsmtSF'].values[0] > 0:
                base_basement = col_e.radio('Basement',['Yes'])
                col_e.markdown(f"Basement Exposure: **{Qual_mapper2[pkl_basehouse['BsmtExposure'].values[0]]}**")
                #reno_Bsmt = col_r.radio('Enhance Basement Exterior Exposure',['No', 'Yes'])
                #if reno_Bsmt == 'Yes':
                    #pkl_renohouse['Exposure'] = 3 
                reno_FinBsmt = col_m.radio('Finish Basement',['No', 'Yes'])
            if reno_FinBsmt == 'Yes':
                pkl_renohouse['FinBsmt_Perc'] = 100
        except:
            col_e.markdown(f"No Basement/Exposure")

        # Garage Quality
        try:
            col_e.markdown(f"Garage Finish: **{Qual_mapper3[pkl_basehouse['GarageFinish'].values[0]]}**")
            reno_Garage = col_m.radio('Finish Garage',['No', 'Yes'])
            if reno_Garage == 'Yes':
                pkl_renohouse['GarageFinish'] = 3 
        except:
            col_e.markdown(f"No Garage")
        
        # Pool
        if pkl_basehouse['PoolArea__1'].values[0] == 0:
            base_pool = col_b.radio('Pool',['No'])
            reno_pool = col_r.radio('Build Pool',['No', 'Yes'])
            pkl_renohouse['PoolArea__1'] = 0 if reno_pool == 'No' else 1
        else:
            base_pool = col_b.radio('Pool',['Yes'])
    
        # Base House MODEL PRICE
        pkl_baseprice = np.floor(np.exp(elasticnet(FinalData, pkl_basehouse)[0]))
        col_bpx.subheader(f'**${num_format(pkl_baseprice)}**')
        col_bpx.caption('Baseline Price Prediction')
        col_bpx.write('-------------------------')
        col_bpx.caption(f"Actual Price: **${num_format(pkl_basehouse['SalePrice'].values[0])}**")
        col_bpx.markdown(f"Livable Space: **{num_format(pkl_basehouse['GrLivArea'].values[0])}** sf")
        col_bpx.markdown(f"Percentage of finished Bsmt: **{num_format(pkl_basehouse['FinBsmt_Perc'].values[0])}** %")
        col_bpx.markdown(f"Garage Size: **{num_format(pkl_basehouse2['GarageCars'].values[0])}** cars")
        col_bpx.markdown(f"finished Outside Spaces: **{num_format(pkl_basehouse['Outside_Spaces'].values[0])}** sf")
        
        # Renovated House PRICE
        pkl_renoprice = np.floor(np.exp(elasticnet(FinalData, pkl_renohouse)[0]))
        col_rpx.subheader(f'**${num_format(pkl_renoprice)}**')
        col_rpx.caption('Renovated House Price')

        # Added metric
        percent_change = round((((pkl_renoprice - pkl_baseprice)/pkl_baseprice)*100),2)
        col_rpx.metric(label='',value='${0}'.format(num_format(pkl_renoprice - pkl_baseprice)),delta='{0}%'.format(percent_change))
        col_rpx.caption('Difference')