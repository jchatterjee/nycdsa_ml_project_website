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



# #loading data
# ## Sidebar Data
avg = load_data('side_data')
# ## Address Data
addID = load_data('address_data')
# ## Model DATA
FinalData = load_data('model_data')
# ## MAP DATA
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
model = pickle.load(open('assets/model.pkl', 'rb'))

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


#=============================================================================

# Sidebar House Selector
# with st.sidebar.container():
    # st.sidebar.title('Model House')

    # sec_select = st.sidebar.selectbox('Select Sector',['North','North West','Downtown','South','South West, South East'])
    # sec_mapper = {'Downtown':'Downtown','South':'South','South West':'South West','South East':'South East','North':'North','North West':'North West'}
    # model_sec = sec_mapper[sec_select]
    # model_neib = st.sidebar.radio('Select Neighborhood',addID.loc[addID.Directions==model_sec]['Neighborhood'].unique())

    # address_df = addID.loc[(addID['Directions']==model_sec) & 
    #     (addID['Neighborhood']==model_neib)]

    # st.sidebar.markdown(f"### {neib_fullname[model_neib]}")
    # try:
    #     st.sidebar.markdown(f"Single-family Detached Houses average price: *${num_format(avg[(avg['BldgType'] =='1Fam') & (avg['Neighborhood'] ==model_neib)]['mean'])}*")
    # except: pass
    # try:
    #     st.sidebar.markdown(f"Two-family Converted Houses average price: *${num_format(avg[(avg['BldgType'] =='2FmCon') & (avg['Neighborhood'] ==model_neib)]['mean'])}*")
    # except: pass
    # try:
    #     st.sidebar.markdown(f"Duplex houses average price: *${num_format(avg[(avg['BldgType'] =='Duplx') & (avg['Neighborhood'] ==model_neib)]['mean'])}*")
    # except: pass
    # try:
    #     st.sidebar.markdown(f"Townhouse End Units average price: *${num_format(avg[(avg['BldgType'] =='TwnhsE') & (avg['Neighborhood'] ==model_neib)]['mean'])}*")
    # except: pass
    # try:
    #     st.sidebar.markdown(f"Townhouse Inside Units average price: *${num_format(avg[(avg['BldgType'] =='Twnhs') & (avg['Neighborhood'] ==model_neib)]['mean'])}*")
    # except: pass




# Page 6 Modeling
if page == "Remodelling":
    with st.container():
        st.header('Remodelling')

        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        sec_select = st.selectbox('Select Sector',['North','North West','Downtown','South','South West, South East'])
        sec_mapper = {'Downtown':'Downtown','South':'South','South West':'South West','South East':'South East','North':'North','North West':'North West'}
        model_sec = sec_mapper[sec_select]
        model_neib = st.radio('Select Neighborhood',addID.loc[addID.Directions==model_sec]['Neighborhood'].unique())

        address_df = addID.loc[(addID['Directions']==model_sec) & (addID['Neighborhood']==model_neib)]

        #st.markdown(f"### {neib_fullname[model_neib]}")


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
            #['Address']

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

            mytable = DataTable(source=source, columns=columns, height=200)
            
            result = streamlit_bokeh_events(
                bokeh_plot=mytable, 
                events="INDEX_SELECT", 
                key="Address", 
                refresh_on_update=True, 
                debounce_time=0,
                override_height=220)

            if result:
                 if result.get("INDEX_SELECT"):
                     # st.markdown(f'#### **{address_df.iloc[result.get("INDEX_SELECT")["data"],0].values[0]}**')
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
        pkl_baseprice = np.floor(model.predict(pkl_basehouse)[0])
        col_bpx.subheader(f'**${num_format(pkl_baseprice)}**')
        col_bpx.caption('Baseline Price Prediction')
        col_bpx.write('-------------------------')
        col_bpx.caption(f"Actual Price: **${num_format(pkl_basehouse['SalePrice'].values[0])}**")
        col_bpx.markdown(f"Livable Space: **{num_format(pkl_basehouse['GrLivArea'].values[0])}** sf")
        col_bpx.markdown(f"Percentage of finished Bsmt: **{num_format(pkl_basehouse['FinBsmt_Perc'].values[0])}** %")
        col_bpx.markdown(f"Garage Size: **{num_format(pkl_basehouse2['GarageCars'].values[0])}** cars")
        col_bpx.markdown(f"finished Outside Spaces: **{num_format(pkl_basehouse['Outside_Spaces'].values[0])}** sf")
        
        # Renovated House PRICE
        pkl_renoprice = np.floor(model.predict(pkl_renohouse)[0])
        col_rpx.subheader(f'**${num_format(pkl_renoprice)}**')
        col_rpx.caption('Renovated House Price')




        # Added metric
        percent_change = round((((pkl_renoprice - pkl_baseprice)/pkl_baseprice)*100),2)
        col_rpx.markdown(f'### **${num_format(pkl_renoprice - pkl_baseprice)}**')
        col_rpx.metric(label='',value='${0}'.format(num_format(pkl_renoprice - pkl_baseprice)),delta='{0}%'.format(percent_change))
        col_rpx.caption('Difference')














