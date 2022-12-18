# CS598-FinalProject
Code for pedestrain safety with commerical headphones. This repository contains the models and a copy of our paper manuscript for our project in improving pedestrian safety using only commerically available devices likely be in-use in the street (i.e. noise cancelling headphones).

## Code Structure

Code is separated into the SVM module and particle filter module. Each one contains separate source directories where the models are written, trained, and saved. Inference is done by loading these individual modules and passing the inference from the SVM to the particle filter along with the audio file tagged to be car positive. The WAV dataset is not uploaded to Github because it takes up too much space for Git to handle (100MB+), but is a combination of our own recordings and the PAWS dataset given by Professor Bashima from UNC.

## Hardware Integration

As outlined in our paper, due to the proprietary firmware in nearly all commercial noise-cancelling headphones, we cannot easily integrate our solution into devices like Airpods. However, our data was collected from these specific headphones for training our models. In addition, hardware integration is possible by attaching two microphones to a RaspberryPi and running the models in inference mode to simulate headphone usage.