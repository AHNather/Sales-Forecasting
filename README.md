# Sales-Forecasting

This project demonstrates the application of various machine learning techniques for sales forecasting. By leveraging historical sales data from Walmart, we aim to predict future sales accurately. The project involves data preprocessing, exploratory data analysis, feature engineering, and the implementation of multiple regression models to identify the most effective approach for sales prediction.

# Project Overview
Sales forecasting is crucial for businesses to manage inventory, plan production, and strategize marketing efforts. Accurate sales predictions help companies optimize resources and make informed decisions. In this project, we use historical sales data provided by Walmart to forecast future sales. The dataset includes information on store sales, features, and store-specific characteristics.

| Field   |      Description      |
|----------|:-------------:|
| Sale-Price |  Sale Price of the property after 5 years from the date of purchase in millions of SAR |
| Purchase-Date | Month and year, when the property was purchased.   |
| Purchase-Price | Property's price at the time of purchase in millions of SAR. |
| Type   |  Type of the property. The property could be open-land, villa, duplex, flat.   |
| Class | Legal classification of the property, could be one of the following options: residential, industrial, or commercial.|
| Location |  Where the property is located w.r.t nearby city. 'Center' (of the city), 'Border' (at entry/exit of city), 'Outskirts' implies on the outskirts of the city. |
| Shape |  Shape of the property. It could be rectangle, trapezoid, irregular.|
| U-Index | Index based on number of utilities available on a scale of 1 to 5. A value of 5 indicates all utilities are available. |
| Proximity | Proximity to the nearest metro station in meters. |
| N-Rank | Rank based on neighborhood facilities that will make the property attractive on a scale of 1 to 10. 1 indicates best neighborhood. |
| P-Chance | Probability of finding parking space on adjacent roads at a given time. It is a value between 0 and 1, where 1 indicates sure availability of parking space. |
| Built | Original year of construction. Applicable for villa, duplex, flat.  |
| Renovate | Latest renovation year. Applicable for villa, duplex, flat. A value of 0 implies no renovation done so far or renovation not applicable. |
| Access |  Type of direct access to the property, which could be street, alley or highway.  |
| Crime-Rate | Average number of crimes reported per year in the neighborhood. |
| C-Rating |  Pleasantness of the climate throughout the year on a scale of 1 to 5. A value of 5 indicates pleasant climate. |
| Gov-Index |  Expected level of government infrastructure project and/or developments in the neighborhood on a scale of 1 to 10. A value of 10 indicates that there are huge developments planned by the government.|
| Contour | Flatness of the property. Applicable only for the open land type property. A value of C indicates the slope of the property is irregular. A value of F indicates the property has a smooth slope. |
| Garage | Is there a private parking garage? Yes or No. Applicable to the flat or duplex type. All villas have private garage. |
| Swimming | Is there a swimming pool? Yes or No. Applicable to the villa type. |
