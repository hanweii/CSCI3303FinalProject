# Food Image Classifier and Food Nutrition Analyzer
**Team member: Hanwei Peng (hp2166), Zijing Sun (zs2198)**

Our goal is to train a food image classifier and implement a food nutrition analyzer.  We firstly use the Food101 dataset (https://www.kaggle.com/kmader/food41) to train a model that is able to recognize food pictures and output the name of food. Next, we will use another dataset called Open Food Facts (https://world.openfoodfacts.org/) to retrieve the nutrition information by providing a food name. Finally, our project will output a visualization of the nutrition details for a given food picture, including a pie chart showing the vitamin percentage and a pie chart showing the micronutrients, as well as a nutrition score and calorie.

This repo have three notebooks: 3303FinalProjectTrainIncV3.ipynb, Demo.ipynb, Resnet50.ipynb.  
One excel file: Nutrition.xlsx.  
One power point file: Project.pptx.  
One image file: hamburger.jpg.  
One InceptionV3 model: InceptionV3_model.hdf5.  
 
**Demo.ipynb** is the notebook for users to upload their food images and get the nutrition details. To use, this notebook need to import two files which are Nutrition.xlsx and InceptionV3_model.hdf5, which are also locate in the repo. Change the file path of the two files if needed, then run each cell in the notebook. At the final cell, change the path of the food image that you want to upload, then you can get its calorie, nutrition score, vitamins pie char and minerals pit char.
 
**3303FinalProjectTrainIncV3.ipynb** is the notebook contains the machine learning lifecycle for training Foof-101 dataset using InceptionV3 model. After training, it will save the model with the lowest val loss as the name "InceptionV3_model.hdf5". To use the model, firstly import load_model ```from keras.models import load_model```, then call
```model = load_model(filepath='./InceptionV3_model.hdf5')```, change the filepath to the location of the model. The detail for using the model is included in the "Test the model" section in this notebook.

**Resnet50.ipynb** is the notebook contains the machine learning lifecycle for training Foof-101 dataset using ResNet50 model. After training, it will save the model with the best val accuracy as the name "model.pth". To use the model, firstly import torch 'import torch', then call 
```
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
model.load_state_dict(torch.load('./model.pth'))
model = V100_model.to('cpu')
```
change the file path to the location of the model. We didn't use ResNet50 model to test because its val accuracy is lower than InceptionV3 model.

**Nutrition.xlsx** is the excel file we recreate base on the Open Food Fact dataset. This excel file take out the nutrition information we want and correspond them with the food name.

**Project.pptx** is our presentation powerpoint.

**hamburger.jpg** is a default picture in Demo.pynb.

**InceptionV3_model.hdf5** is the InceptionV3 model with val accuracy 0.87.




