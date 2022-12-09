# Interactive Data Analytics Dashboard
 
The following website was created in Streamlit to automatically generate a web-based interactive dashboard from Python code. It utilized a data set from 2020 of all the prices of houses and their unique features in order to develop a machine learning prediction model utilizing ElasticNet at more than 93% accuracy.

The following features were used to estimate the average price of a house in a particular area and neighborhood of the town:

1. Number of Rooms
2. Number of Bathrooms
3. Number of Garage Car Spaces
4. Ground Living Area (square feet)
5. Lot Area (square feet)
6. Lot Frontage (linear feet)
7. Overall Quality of House (on a scale from 1 to 9)
8. Approximate Age (years)
9. Presence of Basement
10. Presence of Paved Driveway
11. Whether or Not Remodeled

The following proposed improvements were used to determine how much added value a particular house in a specific area and neighborhood would have if implemented:

1. Remodeling Exterior Material
2. Remodeling Kitchen
3. Building Pool
4. Finishing Basement
5. Finishing Garage

A variety of machine learning regression models such as Multilinear, Logistic, Huber, Penalized, Ridge, Lasso, Random Forest, and Support Vector Machines (SVM) were attempted but ElasticNet turned out to be the most accurate and easily implemented one. In conclusion, a combination of quantitative and ordinal categorical features working together to determine the overall price of a house and its improvements made it ideal for this task.

The fully functioning website can be found here:
https://share.streamlit.io/jchatterjee/nycdsa_ml_project_website/main/trialapp.py

This is the repository for the website of the team consisting of Joydeep Chatterjee, Layal Hammad, Monika Singh, and Stepan Skorkin for the Machine Learning Project (I) on Ames, IA housing prices as part of the 12-week Data Science Bootcamp at NYC Data Science Academy.

The StreamLit website service shall point to this directory in order to view the source code to generate the website.

The full repository of the project code can be found here:
https://github.com/MonikaSinghGit/AmesIowa
