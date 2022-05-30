import os
import streamlit as st
import subprocess
import sys
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

# --- Initialising Path to Run Map Script in Parallel ---
sys.path.append("some_path")
import Ames_map


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

# Loading House Points
with open('assets/house_points.txt') as f:
    house_points = f.readlines()
f.close()
# Loading Map Outline Coordinates
with open('assets/map_outline.txt') as f:
    map_outline = f.readlines()
f.close()

## Sidebar Data
avg = load_data('side_data')
## Address Data
addID = load_data('address_data')
## Model DATA
FinalData = load_data('model_data')
## MAP DATA
merged = load_data('map_data')

# st.write('Contents of the `.streamlit/config.toml` file of this app')

# st.code("""
# [theme]
# primaryColor="#F39C12"
# backgroundColor="#2E86C1"
# secondaryBackgroundColor="#AED6F1"
# textColor="#FFFFFF"
# font="monospace"
# """)

# st.markdown('<link rel="stylesheet" href = "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
# st.markdown("""
# <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #FF4B4B;">
#   <a class="navbar-brand" href="www.youtube.com" target="_blank" MJSL Consulting </a> 
#   <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
#     <span class="navbar-toggler-icon"></span>
#   </button>

#   <div class="collapse navbar-collapse" id="navbarNav">
#     <ul class="navbar-nav">
#       <li class="nav-item active">
#         <a class="nav-link"  disabled" href="youtube.com">Home <span class="sr-only">(current)</span></a>
#       </li>
#       <li class="nav-item">
#         <a class="nav-link" href="youtub.com" target = "_blank"> Price Prediction</a>
#       </li>
#       <li class="nav-item">
#         <a class="nav-link " href="twitter.com" id="navbarDropdown" target = "_blank"> Remodelling</a> 
#       </li>
#       <li class="nav-item">
#         <a class="nav-link disabled" href="#" tabindex="-1" aria-disabled="true">Disabled</a>
#       </li>
#     </ul>
#   </div>
# </nav>
# """, unsafe_allow_html = True)

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

#=============================================================================
# Map of Ames
if page == "Map of Ames":
    with st.container():
        st.title('Map of Ames')
        #st.session_state.load_state = True
        fig = go.Figure(data=[go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Blmngtn']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Blmngtn']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Bloomington Heights', hovertemplate =  "<b>Neighborhood: </b> Bloomington Heights <br>" +"<b>Min Price : </b> 159895 <br>" + "<b>Max Price : </b> 246990 <br>"+"<b>1Fam Avg Price : </b> 177689 <br>" + "<b>TwnhsE Avg Price : </b> 199019 <br>") , go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='BrDale']['Longitude'], lat=merged.loc[merged['Neighborhood']=='BrDale']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Briardale', hovertemplate =  "<b>Neighborhood: </b> Briardale <br>" +"<b>Min Price : </b> 83000 <br>" + "<b>Max Price : </b> 125500 <br>"+"<b>Twnhs Avg Price : </b> 104467 <br>" + "<b>TwnhsE Avg Price : </b> 112333 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='ClearCr']['Longitude'], lat=merged.loc[merged['Neighborhood']=='ClearCr']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Clear Creek', hovertemplate =  "<b>Neighborhood: </b> Clear Creek <br>" +"<b>Min Price : </b> 107500 <br>" + "<b>Max Price : </b> 328000 <br>"+"<b>1Fam Avg Price : </b> 215662 <br>" + "<b>Twnhs Avg Price : </b> 148400 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Gilbert']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Gilbert']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Gilbert', hovertemplate =  "<b>Neighborhood: </b> Gilbert <br>" +"<b>Min Price : </b> 115000 <br>" + "<b>Max Price : </b> 377500 <br>"+"<b>1Fam Avg Price : </b> 189999 <br>" + "<b>2fmCon Avg Price : </b> 150000 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='NAmes']['Longitude'], lat=merged.loc[merged['Neighborhood']=='NAmes']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'North Ames', hovertemplate =  "<b>Neighborhood: </b> North Ames <br>" +"<b>Min Price : </b> 68000 <br>" + "<b>Max Price : </b> 345000 <br>"+"<b>1Fam Avg Price : </b> 146314 <br>" + "<b>2fmCon Avg Price : </b> 132000 <br>" +"<b>Duplex Avg Price : </b> 127547 <br>" + "<b>TwnhsE Avg Price : </b> 167800 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='NPkVill']['Longitude'], lat=merged.loc[merged['Neighborhood']=='NPkVill']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Northpark Villa', hovertemplate =  "<b>Neighborhood: </b> Northpark Villa <br>" +"<b>Min Price : </b> 120000 <br>" + "<b>Max Price : </b> 155000 <br>"+"<b>Twnhs Avg Price : </b> 189999 <br>" + "<b>TwnhsE Avg Price : </b> 150000 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='NWAmes']['Longitude'], lat=merged.loc[merged['Neighborhood']=='NWAmes']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Northwest Ames', hovertemplate =  "<b>Neighborhood: </b> Northwest Ames <br>" +"<b>Min Price : </b> 82500 <br>" + "<b>Max Price : </b> 306000 <br>"+"<b>1Fam Avg Price : </b> 192074 <br>" + "<b>2fmCon Avg Price : </b> 146500 <br>" +"<b>Duplex Avg Price : </b> 137468 <br>" ), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='StoneBr']['Longitude'], lat=merged.loc[merged['Neighborhood']=='StoneBr']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#9190E1'), name = 'Stone Brook', hovertemplate =  "<b>Neighborhood: </b> Stone Brook <br>" +"<b>Min Price : </b> 150000 <br>" + "<b>Max Price : </b> 591587 <br>"+"<b>1Fam Avg Price : </b> 380730 <br>" + "<b>TwnhsE Avg Price : </b> 233313 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Greens']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Greens']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Greensboro', hovertemplate =  "<b>Neighborhood: </b> Greensboro <br>" +"<b>Min Price : </b> 155000 <br>" + "<b>Max Price : </b> 214000 <br>"+"<b>Twnhs Avg Price : </b> 194500 <br>" + "<b>TwnhsE Avg Price : </b> 192562 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='NridgHt']['Longitude'], lat=merged.loc[merged['Neighborhood']=='NridgHt']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Northridge Heights', hovertemplate =  "<b>Neighborhood: </b> Northridge Heights <br>" +"<b>Min Price : </b> 154000 <br>" + "<b>Max Price : </b> 615000 <br>"+"<b>1Fam Avg Price : </b> 354578 <br>" + "<b>Twnhs Avg Price : </b> 191541 <br>" + "<b>TwnhsE Avg Price : </b> 234802 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='NoRidge']['Longitude'], lat=merged.loc[merged['Neighborhood']=='NoRidge']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Northridge', hovertemplate =  "<b>Neighborhood: </b> Northridge <br>" +"<b>Min Price : </b> 190000 <br>" + "<b>Max Price : </b> 755000 <br>"+"<b>1Fam Avg Price : </b> 326114 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Sawyer']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Sawyer']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Sawyer', hovertemplate =  "<b>Neighborhood: </b> Sawyer <br>" +"<b>Min Price : </b> 62383 <br>" + "<b>Max Price : </b> 219000 <br>"+"<b>1Fam Avg Price : </b> 137760 <br>" + "<b>2fmCon Avg Price : </b> 124100 <br>" + "<b>Duplex Avg Price : </b> 139340 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='SawyerW']['Longitude'], lat=merged.loc[merged['Neighborhood']=='SawyerW']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Sawyer West', hovertemplate =  "<b>Neighborhood: </b> Sawyer West <br>" +"<b>Min Price : </b> 67500 <br>" + "<b>Max Price : </b> 320000 <br>"+"<b>1Fam Avg Price : </b> 191220 <br>" + "<b>Duplex Avg Price : </b> 211421 <br>" + "<b>TwnhsE Avg Price : </b> 146327 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Somerst']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Somerst']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Somerset', hovertemplate =  "<b>Neighborhood: </b> Somerset <br>" +"<b>Min Price : </b> 139000 <br>" + "<b>Max Price : </b> 468000 <br>"+"<b>1Fam Avg Price : </b> 251257 <br>" + "<b>Twnhs Avg Price : </b> 174018 <br>" + "<b>TwnhsE Avg Price : </b> 201277 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Veenker']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Veenker']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#B3DECD'), name = 'Veenker', hovertemplate =  "<b>Neighborhood: </b> Veenker <br>" +"<b>Min Price : </b> 139000 <br>" + "<b>Max Price : </b> 385000 <br>"+"<b>1Fam Avg Price : </b> 246797 <br>" + "<b>TwnhsE Avg Price : </b> 267340 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='BrkSide']['Longitude'], lat=merged.loc[merged['Neighborhood']=='BrkSide']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#F5F2A1'), name = 'Brookside', hovertemplate =  "<b>Neighborhood: </b> Brookside <br>" +"<b>Min Price : </b> 39300 <br>" + "<b>Max Price : </b> 223500 <br>"+"<b>1Fam Avg Price : </b> 126079 <br>" + "<b>2fmCon Avg Price : </b> 123500 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='IDOTRR']['Longitude'], lat=merged.loc[merged['Neighborhood']=='IDOTRR']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#F5F2A1'), name = 'Iowa DOT and Rail Road', hovertemplate =  "<b>Neighborhood: </b> Iowa DOT and Rail Road <br>" +"<b>Min Price : </b> 34900 <br>" + "<b>Max Price : </b> 212300 <br>"+"<b>1Fam Avg Price : </b> 109321 <br>" + "<b>2fmCon Avg Price : </b> 95996 <br>" +"<b>Duplex Avg Price : </b> 110000 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Landmrk']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Landmrk']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#F5F2A1'), name = 'Landmark Villas', hovertemplate =  "<b>Neighborhood: </b> Landmark Villas <br>" +"<b>Min Price : </b> 137000 <br>" + "<b>Max Price : </b> 137000 <br>"+"<b>Twnhs Avg Price : </b> 137000 <br>" ), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='OldTown']['Longitude'], lat=merged.loc[merged['Neighborhood']=='OldTown']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#F5F2A1'), name = 'Old Town', hovertemplate =  "<b>Neighborhood: </b> Old Town <br>" +"<b>Min Price : </b> 12789 <br>" + "<b>Max Price : </b> 475000 <br>"+"<b>1Fam Avg Price : </b> 127045 <br>" + "<b>2fmCon Avg Price : </b> 127368 <br>" +"<b>Duplex Avg Price : </b> 121100 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='GrnHill']['Longitude'], lat=merged.loc[merged['Neighborhood']=='GrnHill']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'Greens Hills', hovertemplate =  "<b>Neighborhood: </b> Greens Hills <br>" +"<b>Min Price : </b> 230000 <br>" + "<b>Max Price : </b> 330000 <br>"+"<b>TwnhsE Avg Price : </b> 280000 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Blueste']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Blueste']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'Bluestem', 
hovertemplate = "<b>Neighborhood: </b> Bluestem <br>" +"<b>Min Price : </b> 115000 <br>" + "<b>Max Price : </b> 200000 <br>"+"<b>Twnhs Avg Price : </b> 1125480 <br>" + "<b>TwnhsE Avg Price : </b> 161700 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='CollgCr']['Longitude'], lat=merged.loc[merged['Neighborhood']=='CollgCr']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'College Creek', hovertemplate =  "<b>Neighborhood: </b> College Creek <br>" +"<b>Min Price : </b> 110000 <br>" + "<b>Max Price : </b> 475000 <br>"+"<b>1Fam Avg Price : </b> 201432 <br>" + "<b>Duplex Avg Price : </b> 185000 <br>"+"<b>TwnhsE Avg Price : </b> 142803 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Crawfor']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Crawfor']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'Crawford', hovertemplate =  "<b>Neighborhood: </b> Crawford <br>" +"<b>Min Price : </b> 90350 <br>" + "<b>Max Price : </b> 392500 <br>"+"<b>1Fam Avg Price : </b> 198512 <br>" + "<b>2fmCon Avg Price : </b> 148500 <br>"+"<b>Duplex Avg Price : </b> 177500 <br>" + "<b>TwnhsE Avg Price : </b> 283083 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Edwards']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Edwards']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'Edwards', hovertemplate =  "<b>Neighborhood: </b> Edwards <br>" +"<b>Min Price : </b> 35000 <br>" + "<b>Max Price : </b> 415000 <br>"+"<b>1Fam Avg Price : </b> 134800 <br>" + "<b>2fmCon Avg Price : </b> 106380 <br>"+"<b>Duplex Avg Price : </b> 121108 <br>" + "<b>Twnhs Avg Price : </b> 132875 <br>" +"<b>TwnhsE Avg Price : </b> 140062 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='SWISU']['Longitude'], lat=merged.loc[merged['Neighborhood']=='SWISU']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'South & West of ISU', hovertemplate =  "<b>Neighborhood: </b> South & West of ISU <br>" +"<b>Min Price : </b> 60000 <br>" + "<b>Max Price : </b> 197000 <br>"+"<b>1Fam Avg Price : </b> 133739 <br>" + "<b>2fmCon Avg Price : </b> 123860 <br>"+"<b>Duplex Avg Price : </b> 155000 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Timber']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Timber']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('#DF8977'), name = 'Timberland', hovertemplate =  "<b>Neighborhood: </b> Timberland <br>" +"<b>Min Price : </b> 150000 <br>" + "<b>Max Price : </b> 425000 <br>"+"<b>1Fam Avg Price : </b> 243189 <br>" + "<b>2fmCon Avg Price : </b> 228950 <br>"+"<b>TwnhsE Avg Price : </b> 242750 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='MeadowV']['Longitude'], lat=merged.loc[merged['Neighborhood']=='MeadowV']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('pink'), name = 'Meadow Village', hovertemplate =  "<b>Neighborhood: </b> Meadow Village <br>" +"<b>Min Price : </b> 73000 <br>" + "<b>Max Price : </b> 151400 <br>"+"<b>Twnhs Avg Price : </b> 89299 <br>" + "<b>TwnhsE Avg Price : </b> 104373 <br>"), go.Scattermapbox(lon=merged.loc[merged['Neighborhood']=='Mitchel']['Longitude'], lat=merged.loc[merged['Neighborhood']=='Mitchel']['Latitude'],mode='markers',marker= go.scattermapbox.Marker(size=9), opacity=0.5, marker_color=('pink'), name = 'Mitchell', hovertemplate =  "<b>Neighborhood: </b> Mitchell <br>" +"<b>Min Price : </b> 81500 <br>" + "<b>Max Price : </b> 300000 <br>"+"<b>1Fam Avg Price : </b> 164066 <br>" + "<b>2fmCon Avg Price : </b> 168000 <br>"+"<b>Duplex Avg Price : </b> 152940 <br>" + "<b>Twnhs Avg Price : </b> 164000 <br>" +"<b>TwnhsE Avg Price : </b> 156200 <br>")]).update_layout(plot_bgcolor='white',autosize=False, width = 1000, height = 800, legend=dict(x=1.0, y=1)).update_yaxes(showgrid=False).update_xaxes(showgrid=False)

        fig.update_layout(autosize=True, hovermode='closest',mapbox=dict( accesstoken="pk.eyJ1IjoibGF5YWxoYW1tYWQiLCJhIjoiY2wzbWd5ZWxjMDFqNDNmcWt5MzRzNHdlaCJ9.B04CAbFi5Llmk2B78EP6JQ", bearing=0,center=dict(lat=42.027269,lon= -93.611500), pitch=0,zoom=11.4,layers = [{'source': { "type":"GeometryCollection","geometries":[{"type":"MultiPolygon","coordinates": [[[map_outline]]]}]}, 'type': "line", 'below': "traces", 'color' :'#7392DA', 'opacity': 0.5}]))
        # Plot!
        st.plotly_chart(fig, use_container_width=True)

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
            
            # Attempt to add drop-down menu, didn't work
            # ['Address']
            # source = col_main.selectbox('Select your Address', address_df)
            # sourceindex = source.selected.index
            # st.markdown(f'sourceindex')

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


        # hstype_mapper = {'1Fam':'Single-family Detached House', '2FmCon':'Two-family Converted Houses',
        #                 'Duplx':'Duplex house', 
        #                 'TwnhsE':'Townhouse End Unit', 'Twnhs':'Townhouse Inside Unit'}
        # col_main.caption(f"{hstype_mapper[pkl_basehouse2['BldgType'].values[0]]} in {model_neib}")

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
            if reno_Kitchen == 'Yes':
                pkl_renohouse['ExterQual'] = 5
        except:
            col_b.markdown(f"Kitchen Quality: **None**")

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
        col_rpx.markdown(f'### **${num_format(pkl_renoprice - pkl_baseprice)}**')
        col_rpx.metric(label='',value='${0}'.format(num_format(pkl_renoprice - pkl_baseprice)),delta='{0}%'.format(percent_change))
        col_rpx.caption('Difference')