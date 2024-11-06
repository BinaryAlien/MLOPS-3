**Part 0**

1. Install locally MLFlow on your setup. For that you can use the method you want (manual install or using docker official image : https://mlflow.org/docs/latest/docker.html)
2. Make sure you can access the web UI of ML flow

**Part 1 - Tracking a model training**

1.1 Using the MLflow documentation (here for instance), make a model training script on a simple dataset of your choice.

The model training script should track :

- the the model hyper-parameters
- the model metric (mse, accuracy or whatever)

No need to version the model for now

1.2 Run the script and check that the tracked data are available in mlflow UI

1.3 Change one hyper parameter of the model and rerun the training script. Check that both this run and the previous one are availables.

1.4 Make sure that the model is also save in MLFlow. Rerun the training script and access it in the interface

**Part 2 - Deploying model from MLFlow**

We now want to have a web service service to serve our machine learning model.

The web service should

- load the model model from the MLFlow server upon start of the server
- a /predict endpoint to return predictions upon a Post request
- a /update-model endpoint allowing to update the model with a webflow model version

1. Build such service with fastAPI or flask, or any other service
2. make a script to test automatically test your /predict endpoint
3. update the script so that it also test that /update-model works well
4. Dockerise your webservice

Remark : the model should not be saved in the Dockerfile since it is loaded by the service from mlflow

**Part 3 - Canary deployment**

We would like now to do canary deployment.

For that we will have two loaded models :

- current (for the current model version)
- next (for the next version we would like to have)

At startup both current and next model should be the same.

the /predict version should use the current model with a p probability and the next model with 1 - p.

the /update-model endpoint should update the next model

the /accept-next-model endpoint should set next-model as current so that both current and next are used
