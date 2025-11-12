CarbonTrackCLI – Command-Line Tool for Tracking, Forecasting, and VIsualizing Carbon Emissions 
Sean Morris 
Problem: 
Climate change mitigation requires accurate, real-time tracking of carbon emissions across countries and main driving sectors of GHG emissions (transportation, energy, etc.). However, most open datasets are fragmented across agencies, and existing dashboards are web-based, not developer-friendly. 
Goal: 
Build a lightweight, open-source CLI (Command-Line Interface) tool that aggregates emissions data from public sources, visualizes trends, and uses machine learning to forecast short-term CO₂ emissions trajectories. This project aims to make emissions analytics accessible to researchers, data scientists, and developers—enabling rapid climate data exploration directly from the terminal. 
Data Collection & Refinement 
Sources: 
EPA Greenhouse Gas Reporting Program (GHGRP) – provides U.S.-specific, sector-level emissions data across major categories such as Energy, Transportation, Industry, Agriculture, and Waste.
World Bank Climate Data – offers country-level economic and demographic variables including GDP, population, and energy use, allowing normalization of emissions per capita and per GDP.
Our World in Data (OWID) – includes comprehensive global datasets on CO₂ and GHG emissions by country and sector, serving as the project’s main longitudinal emissions source.
Refinement Steps: 
Combine datasets into a standardized schema: (country, year, sector, emissions_co2e, population, gdp).
Map all sector labels into a unified taxonomy consistent with IPCC’s five major categories: Energy, Industry, Transportation, Agriculture, and Waste.
Perform data cleaning and missing-value imputation with pandas; cache standardized data locally to support offline CLI use.
Implementation 
Pipeline Overview 
1. Ingestion Layer 
a. Commands to fetch and cache latest data using APIs or CSV curls 
Example:
carbontrack fetch –source owid
Carbontrack fetch –source epa
2. Preprocessing Layer 
a. Python-based ETL using pandas
3. Modeling Layer 
a. XGBoost Regressor to forecast next 5 years of CO2 emissions for a given country and/or sector 
b. Features include GDP, population, historical emissions, and sectoral intensity metrics.
c. The executable will be written in Python, allowing seamless integration of the trained model with the CLI package.
4. Inference Layer 
Users will have two inference modes:
Pre-computed forecasts: the CLI retrieves cached 5-year predictions for all countries/sectors.
On-demand inference: the CLI runs real-time predictions using the embedded XGBoost model when requested.
 b. Commands to visualize trends and forecasts:
Evaluation
Model Metrics
Mean Squared Error (MSE)
Mean Absolute Percentage Error (MAPE)


Usability Metrics
CLI latency (command/pipeline execution time)
Ease of installation and runtime reproducibility


Sustainability Metrics
The project will measure and report its own computational carbon footprint using the CodeCarbon library.
This will track CO₂e emissions generated during model training and inference, ensuring alignment between project goals and sustainable computing practices.
Summary

CarbonTrackCLI provides a transparent, developer-accessible platform for analyzing and forecasting carbon emissions. By combining open data sources, a reproducible Python pipeline, and a command-line interface, this project bridges data science and climate awareness—enabling efficient, low-friction environmental analytics directly from the terminal.
