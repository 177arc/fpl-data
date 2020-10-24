![](https://github.com/177arc/fpl-data/workflows/CI%2FCD/badge.svg)

# AWS lambda function for preparing FPL data
The purpose of this project is provide to an [AWS lambda function](https://aws.amazon.com/lambda/) that retrieves data from the FPL API, processes it and makes the prepared data sets available for further data analysers
such as the [FPL Advisor](https://github.com/177arc/fpl-advisor). The data sets are published in a public S3 bucket: https://s3.eu-west-2.amazonaws.com/fpl.177arc.net/

The lambda function runs in AWS on an hourly schedule during the day and continously updates the data. The different processing steps are co-ordinated via the 
[prep-data.ipynb Jupyter notebook](https://github.com/177arc/fpl-data/blob/develop/prep_data.ipynb).

