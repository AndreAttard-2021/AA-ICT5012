The following repository ('AA-ICT5012') contains all of the code, data and results relevant to the dissertation entitled "Route Optimisation of the Public Transport Network in Malta Using Reinforcement Learning" submitted in partial fulfilment of the requirements for the degree of Master of Science in Data Science at the University of Malta.

An explanation of each folder is as follows:

Section 3.1 - Contains the Jupyter notebooks used to carry out the following tasks:
1) Web Scrap information from the MPT Website ('Scraping_Route_Data.ipynb'). Data was originally scraped from the MPT website (https://www.publictransport.com.mt/routes-timetables-view-all-routes/) in November 2024. However as of May 2025, the MPT website has undergone significant changes, hence the code may not function as intended.
2) Obtain Longitude and Latitude data of bus stops using Geocoding services ('Geocoding.ipynb'). The personal Google Cloud API Key has been removed for security reasons, however a new API key can be obtained by setting up an account through the following link: https://mapsplatform.google.com/
3) Obtain in-vehicle Travel Times between bus stop connections, generate inputs of the Bus Network Design Problem and analyse the data. The personal TomTom API Key has been removed for security reasons, however a new API key can be obtained by setting up an account through the following link: https://www.tomtom.com/products/routing-apis/

Section 3.2 - Contains the code used to run the Branch-and-Cut algorithm ('GUROBI_BnC_Code.py'). The details related to the personal WLS licence have been removed. Kindly vist the following site to learn more about Gurobi licensing and to set-up an academic licence: https://portal.gurobi.com/iam/login/?target=https%3A%2F%2Fportal.gurobi.com%2F 

To run the code, the following command-line argument can be utilised in a Jupyter notebook:

!python "/content/drive/Othercomputers/My laptop/ICT5012 - Disseration/GUROBI Model/GUROBI_BnC_Code.py"  \
  --filename "/content/drive/Othercomputers/My laptop/ICT5012 - Disseration/transit_learning-master/CEC2013Supp/Instances/Morning_Model/GozoTravelTimes.txt" \
  --terminals [0,15,21,22,23,38,40,47,82,107,137,164,187,191,203,216] \
  --L 42 --q 11 --m_max 15 --time_limit 14400 --seed 23

Note that paths should be adjusted accordingly:
1) The filename can be obtained from the following path 'Section 3.3/CEC2013Supp/Instances/Morning_Model' (Indicate the TravelTimes.txt file of the required experiment)
2) List of terminals can be obtained from 'AA-ICT5012/Section 3.1/Datasets/Files used for Data Visualisation/Routes/Routes_Working/'.
3) L is equivalent to maximum number of stops one route can visit
4) q is the minimum number of stops.
5) m_max is the number of routes.
6) time_limit denotes the maximum running time of BnC algorithm
7) seed is the seed number

The outputs for all experiments covered in dissertation are available in 'Section 3.2/BnC Results'

Section 3.3 - Contains the code use to run the Reinforcement Learning Algorithm. 

To train the model, the following command-line argument can be utilised in a Jupyter notebook (After installing all necessary python libraries as noted in 'Section 3.3/cc_requirements.txt'): 

!python "/content/drive/Othercomputers/My laptop/ICT5012 - Disseration/transit_learning-master/learning/inductive_route_learning.py" \
    eval.dataset.path="/content/drive/Othercomputers/My laptop/ICT5012 - Disseration/transit_learning-master/CEC2013Supp/Instances/Evening_Model" \
    +eval=mater_dei_south_routes +run_name=MaterDeiSouthRoutes_Evening_TerminalNodes

Note that paths should be adjusted accordingly (In addition, paths within the Python scripts being called by command-line argument need to be adjusted accordingly)

All weights obtained for each experiment in this dissertation are available in 'Section 3.3/output'

To analyse training progression using Tensorboard, the following command-line argument can be utilised in a Jupyter notebook:

%load_ext tensorboard
%tensorboard --logdir "/content/drive/Othercomputers/My laptop/ICT5012 - Disseration/transit_learning-master/training_logs/inductive_MaterDeiSouthRoutes_Evening_TerminalNodes"

Note that paths should be adjusted accordingly (In addition, paths within the Python scripts being called by command-line argument need to be adjusted accordingly)

All Tensorboard outpus for each experiment in this dissertation are availabe in 'Section 3.3/training_logs'

To evaluate the model, the following command-line argument can be utilised in a Jupyter notebook:

!python "/content/drive/Othercomputers/My laptop/ICT5012 - Disseration/transit_learning-master/learning/eval_route_generator.py" \
    +model.weights="/content/drive/Othercomputers/My laptop/ICT5012 - Disseration/transit_learning-master/output/inductive_MaterDeiSouthRoutes_Evening_TerminalNodes.pt" \
    eval.dataset.path="/content/drive/Othercomputers/My laptop/ICT5012 - Disseration/transit_learning-master/CEC2013Supp/Instances/Evening_Model" \
    +eval=mater_dei_south_routes +run_name=MaterDeiSouth_Evening_TerminalNodes_Evaluation

Note that paths should be adjusted accordingly (In addition, paths within the Python scripts being called by command-line argument need to be adjusted accordingly)

All optimal routes obtained for the experiments carried out in this dissertation are saved as pickle files in the following path: 'Section 3.3/output_routes'

Chapter 4 - Contains the Jupyter notebook used to calculate the metric values for the rotues used in Malta's Public Transportation system and the results obtained using the Branch and Cut algorithm.
