This repository accompanies the paper Preserving Sovereignty and Privacy for Personalization: Designing a Federated Recommendation System for Data Spaces


---

Included Files

The `data.ipynb` file includes the data preprocessing and descriptive statitics of the data


The `dmf.py` includes the the traditional deep matrix factoriaztion model and the logic for localized and centralized training



The `scenario_local.ipynb` includes the training and evaluation of the local sceanrio, where each participants train the model solely on its own data 

The `scenario_centralized.ipynb` includes the training and evaluation of the centralized sceanrio, where all participants pool the training data in one single repository and train a holistic model without privacy considerations

The `scenario_fedrec.ipynb` includes the training and evaluation of the federated sceanrio, where participants use the propsoed solution to train a recommendation model in a federated manner


The `scenario_fedrec_bandwidth.ipynb` shows the bandwidth of the federated sceanrio


