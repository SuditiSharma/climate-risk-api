# Known Limitations

## 1. Circular Label Problem
The risk label was engineered from deaths, affected populations, 
economic damages and physical magnitude. These same variables later 
dominated feature importance at over 80% combined. This means the 
model partially rediscovers the label formula rather than learning 
risk independently.

## 2. Missing Contextual Variables
The real drivers of property risk are not in this dataset:
- Population density
- Building stock and construction quality
- Infrastructure resilience
- Geospatial and land use context

## 3. Static Model
The model is trained on historical data from 1970 to 2021. 
Risk patterns change as climate changes. A production system 
would need regular retraining on recent data.

## 4. Label Subjectivity
The weightings used to construct the risk label (deaths 40%, 
affected 30%, damages 20%, magnitude 10%) are a design decision 
not a ground truth. Different weightings would produce different 
risk classifications.

## Future Work
A more robust model would combine this dataset with:
- Satellite imagery
- Population and census data
- Building stock databases
- Real-time climate monitoring data

This is the direction multimodal data fusion research addresses.
