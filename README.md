<img src="https://i.imgur.com/HFM4jZT.png" width=150 height=150 align="center" />

# BostonCracks

BostonCracks is a predictive model of the many cracks in Boston's sidewalks. Seeking to reduce the discrepancy in sidewalk repairs completed in neighborhoods of different median incomes, BostonCracks makes use of open data from the city of Boston to more equitably direct the city's resources to sidewalks in need.

BostonCracks is built using keras and pytorch.

<img src="https://i.imgur.com/8X1rFTY.png" />


## Models

BostonCracks takes advantage of two different types of models: visual and statistical.

The visual model utilizes image data scraped from Google Street View. It uses a separate model trained on driverless cars to locate the sidewalk in the image, and then determines the condition of the sidewalk from the processed image. Images of sidewalks in various conditions are also presented to the users of the BostonCracks app for classification, to expand the dataset on which the model is trained.

The statistical model examines the geographical data of a certain piece of sidewalk, using that area's concentration of trees, median income, the number of times the sidewalk has been reported as cracked, and the concentration of trees in the area to determine the likelihood that the sidewalk is cracked.

City employees can use the recommendations from both models to determine where to send workers to examine and repair sidewalk cracks.

### Usage

This repository contains descriptions of data and models used to predict sidewalk cracking. Clone the repo to evaluate or tinker with the models.


[logo]: https://github.com/samc24/BostonCracks/blob/master/logo.png
