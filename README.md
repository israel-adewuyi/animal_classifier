# Animal Classifier Deployment

This project demonstrates how to deploy a Convolutional Neural Network (CNN) model using FastAPI for the API and Streamlit for the web application. The model is trained on a filtered subset of the CIFAR-10 dataset, which classifies three different animal classes: cats, dogs, and horses. The project is containerized using Docker and orchestrated using docker-compose.

# Project Structure

├── code
│   ├── datasets
│   │   └── animal_dataset.py        # Script to generate the animal dataset
│   ├── deployment
│   │   ├── api
│   │   │   ├── Dockerfile           # Dockerfile for the FastAPI service
│   │   │   └── main.py              # FastAPI code for serving the model
│   │   └── app
│   │       ├── Dockerfile           # Dockerfile for the Streamlit app
│   │       └── app.py               # Streamlit app for user interface
│   └── models
│       └── train_model.py           # Script for training the animal classifier model
├── data                             # Directory for any additional data
├── models                           # Directory to store the trained model (e.g., animal_classifier.pth)
└── docker-compose.yml               # docker-compose file to manage the FastAPI and Streamlit services


# Getting Started

## Prerequisites

To run this project, you will need to have the following installed on your machine:

Docker
Docker Compose
Git
Steps to Run the Project
Clone the repository:

bash
Copy code
git clone <repository_url>
cd <repository_folder>
Train the model:

Before deploying the model, you'll need to train it. Navigate to the code/models/ directory and run the training script:

bash
Copy code
python code/models/train_model.py
This will train a CNN model on the dataset and save the model as animal_classifier.pth in the models/ folder.

Build and Run the Containers:

Once the model is trained, you can build and run the Docker containers:

bash
Copy code
docker-compose up --build
This command will start the following services:

API Service: A FastAPI server running on http://localhost:8000, which serves the model for predictions.
App Service: A Streamlit web application running on http://localhost:8501, which provides a UI to upload images and get predictions.
Interact with the Web Application:

After the containers are running, open your browser and navigate to:

arduino
Copy code
http://localhost:8501
Here, you can upload images of animals (cats, dogs, and horses) and receive predictions from the deployed CNN model.
