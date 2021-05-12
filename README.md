# Food Image Classifier and Food Nutrition Analyzer
**Team member: Hanwei Peng (hp2166), Zijing Sun (zs2198)**

Our goal is to train a food image classifier and implement a food nutrition analyzer.  We firstly use the Food101 dataset (https://www.kaggle.com/kmader/food41) to train a model that is able to recognize food pictures and output the name of food. Next, we will use another dataset called Open Food Facts (https://world.openfoodfacts.org/) to retrieve the nutrition information by providing a food name. Finally, our project will output a visualization of the nutrition details for a given food picture, including a pie chart showing the vitamin percentage and a pie chart showing the micronutrients, as well as a nutrition score and calorie.

This repo have four notebooks: 3303FinalProjectTrainIncV3.ipynb, Demo.ipynb, Resnet50.ipynb, PredictOneKind.ipynb.  
One excel file: Nutrition.xlsx.  
One power point file: Project.pptx.  
One image file: hamburger.jpg.    
 
**Demo.ipynb** is the notebook for users to upload their food images and get the nutrition details. To use, this notebook need to import two files which are Nutrition.xlsx and InceptionV3_model.hdf5. Nutrition.xlsx is in the repo. To get InceptionV3_model.hdf5, one way is to download the model we trained from this google drive link: https://drive.google.com/file/d/1H8Lpi94cYoiPtHJGvj81PvDnUGiomXZ5/view?usp=sharing (since the model is too big to upload to github) and this model has accuracy about 0.87 another way is to run **3303FinalProjectTrainIncV3.ipynb** to train another model. Change the file path of the two files if needed, then run each cell in the notebook. At the final cell, change the path of the food image that you want to upload, then you can get its calorie, nutrition score, vitamins pie char and minerals pit char.
 
**3303FinalProjectTrainIncV3.ipynb** is the notebook contains the machine learning lifecycle for training Food-101 dataset using InceptionV3 model. After training, it will save the model with the lowest val loss as the name "InceptionV3_model.hdf5". To use the model, firstly import load_model ```from keras.models import load_model```, then call
```model = load_model(filepath='./InceptionV3_model.hdf5')```, change the filepath to the location of the model. The detail for using the model is included in the "Test the model" section in this notebook:
```
from PIL import Image
from keras.preprocessing import image

test_image = Image.open('food-101/test_60/waffles/1005755.jpg')
img = image.load_img('food-101/test_60/waffles/1005755.jpg', target_size=(299, 299))
img_arr = image.img_to_array(img)
img_arr = np.expand_dims(img_arr, axis=0)
img_arr_test = preprocess_img(img_arr)
y_pred = model.predict(img_arr_test)
preds = np.argmax(y_pred, axis=1)
best_pred = collections.Counter(preds).most_common(1)[0][0]
print(arr[best_pred])
plt.imshow(test_image)
```

**Resnet50.ipynb** is the notebook contains the machine learning lifecycle for training Food-101 dataset using ResNet50 model. After training, it will save the model with the best val accuracy as the name "model.pth". To use the model, firstly import torch 'import torch', then call 
```
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
model.load_state_dict(torch.load('./model.pth'))
model = V100_model.to('cpu')
```
change the file path to the location of the model. We didn't use ResNet50 model to test because its val accuracy is lower than InceptionV3 model.

**PredictOneKind.ipynb** is the notebook that gives the prediction accuracy for each type of food.

**Nutrition.xlsx** is the excel file we recreate base on the Open Food Fact dataset. This excel file take out the nutrition information we want and correspond them with the food name.

**Project.pptx** is our presentation powerpoint.

**hamburger.jpg** is a default picture in Demo.pynb.

**InceptionV3_model.hdf5** is the InceptionV3 model with val accuracy 0.87. It can be download at https://drive.google.com/file/d/1H8Lpi94cYoiPtHJGvj81PvDnUGiomXZ5/view?usp=sharing




