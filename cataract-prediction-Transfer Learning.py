import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
import numpy as np 
import pandas as pd 
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.vgg19 import VGG19,EfficientNetB7
from tensorflow.keras.applications import EfficientNetB7,InceptionV3,ResNet152,InceptionResNetV2,Xception
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from plot_keras_history import show_history, plot_history




df = pd.read_csv("C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/Ocular Disease Recognition/full_df.csv")
df.head(3)
def has_cataract(text,kind):
    if kind in text:
        return 1
    else:
        return 0

df["left_cataract"] = df["Left-Diagnostic Keywords"].apply(lambda x: has_cataract(x,'cataract'))
df["right_cataract"] = df["Right-Diagnostic Keywords"].apply(lambda x: has_cataract(x,'cataract'))
left_cataract = df.loc[(df.C ==1) & (df.left_cataract == 1)]["Left-Fundus"].values
right_cataract = df.loc[(df.C ==1) & (df.right_cataract == 1)]["Right-Fundus"].values
print("Number of images in left cataract: {}".format(len(left_cataract)))
print("Number of images in right cataract: {}".format(len(right_cataract)))

left_normal = df.loc[(df.C ==0) & (df["Left-Diagnostic Keywords"] == "normal fundus")]["Left-Fundus"].sample(250,random_state=42).values
right_normal = df.loc[(df.C ==0) & (df["Right-Diagnostic Keywords"] == "normal fundus")]["Right-Fundus"].sample(250,random_state=42).values
cataract = np.concatenate((left_cataract,right_cataract),axis=0)
normal = np.concatenate((left_normal,right_normal),axis=0)
print(len(cataract),len(normal))

dataset_dir = "C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/Ocular Disease Recognition/preprocessed_images/"
image_size=224
labels = []
dataset = []
def create_dataset(image_category,label):
    for img in tqdm(image_category):
        image_path = os.path.join(dataset_dir,img)
        try:
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
            image = cv2.resize(image,(image_size,image_size))
        except:
            continue
        dataset.append([np.array(image),np.array(label)])
    random.shuffle(dataset)
    return dataset
        
dataset = create_dataset(cataract,1)
dataset = create_dataset(normal,0)

plt.figure(figsize=(12,7))
plt.suptitle('Show original data',fontsize=30)
for i in range(10):
    sample = random.choice(range(len(dataset)))
    image = dataset[sample][0]
    category = dataset[sample][1]
    if category== 0:
        label = "Normal"
    else:
        label = "Cataract"
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel(label)
plt.tight_layout()    
plt.savefig('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Show original data.png')

x = np.array([i[0] for i in dataset]).reshape(-1,image_size,image_size,3)
y = np.array([i[1] for i in dataset])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


vgg19 = VGG19(weights="imagenet",include_top = False,input_shape=(image_size,image_size,3))
for layer in vgg19.layers:
    layer.trainable = False
model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
w_path='C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/logs/'
checkpoint = ModelCheckpoint(w_path+"vgg19.h5",monitor="val_acc",verbose=1,save_best_only=True,
                             save_weights_only=False,period=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
history = model.fit(x_train,y_train,batch_size=32,epochs=50,validation_data=(x_test,y_test)
            ,callbacks=[checkpoint,earlystop])

loss,accuracy = model.evaluate(x_test,y_test)
print("loss:",loss)
print("Accuracy:",accuracy)

y_pred = model.predict_classes(x_test)

show_history(history)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Training_history VGG19.png",
             title="Training history VGG19")

model.save('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/model/'+'VGG19_eyedisease.h5')
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat = cm,figsize=(8,7),class_names = ["Normal","Cataract"],show_normed = True);
plt.savefig('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/VGG19 predict confusion matrix.png')                 


plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]
    if category== 0:
        label = "Normal"
    else:
        label = "Cataract"      
    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "Cataract"   
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout() 
plt.suptitle("Predict show VGG19",fontsize=30)
plt.savefig("C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Predict show VGG19.png")






efficientNetB7 = EfficientNetB7(weights="imagenet",include_top = False,input_shape=(image_size,image_size,3))
for layer in efficientNetB7.layers:
    layer.trainable = False
model = Sequential()
model.add(efficientNetB7)
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
w_path='C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/logs/'
checkpoint = ModelCheckpoint(w_path+"efficientNetB7.h5",monitor="val_acc",verbose=1,save_best_only=True,
                             save_weights_only=False,period=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
history = model.fit(x_train,y_train,batch_size=32,epochs=50,validation_data=(x_test,y_test)
            ,callbacks=[checkpoint,earlystop])

loss,accuracy = model.evaluate(x_test,y_test)
print("loss:",loss)
print("Accuracy:",accuracy)

y_pred = model.predict_classes(x_test)

show_history(history)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Training_history efficientNetB7.png",
             title="Training history efficientNetB7")

model.save('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/model/'+'efficientNetB7_eyedisease.h5')
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat = cm,figsize=(8,7),class_names = ["Normal","Cataract"],show_normed = True);
plt.savefig('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/efficientNetB7 predict confusion matrix.png')                 


plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]
    if category== 0:
        label = "Normal"
    else:
        label = "Cataract"      
    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "Cataract"   
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout() 
plt.suptitle("Predict show efficientNetB7",fontsize=30)
plt.savefig("C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Predict show efficientNetB7.png")




inceptionV3 = InceptionV3(weights="imagenet",include_top = False,input_shape=(image_size,image_size,3))
for layer in inceptionV3.layers:
    layer.trainable = False
model = Sequential()
model.add(inceptionV3 )
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
w_path='C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/logs/'
checkpoint = ModelCheckpoint(w_path+"inceptionV3.h5",monitor="val_acc",verbose=1,save_best_only=True,
                             save_weights_only=False,period=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
history = model.fit(x_train,y_train,batch_size=32,epochs=50,validation_data=(x_test,y_test)
            ,callbacks=[checkpoint,earlystop])

loss,accuracy = model.evaluate(x_test,y_test)
print("loss:",loss)
print("Accuracy:",accuracy)

y_pred = model.predict_classes(x_test)

show_history(history)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Training_history inceptionV3.png",
             title="Training history inceptionV3")

model.save('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/model/'+'inceptionV3_eyedisease.h5')
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat = cm,figsize=(8,7),class_names = ["Normal","Cataract"],show_normed = True);
plt.savefig('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/inceptionV3 predict confusion matrix.png')                 


plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]
    if category== 0:
        label = "Normal"
    else:
        label = "Cataract"      
    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "Cataract"   
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout() 
plt.suptitle("Predict show inceptionV3 ",fontsize=30)
plt.savefig("C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Predict show inceptionV3.png")



resNet152=ResNet152(weights="imagenet",include_top = False,input_shape=(image_size,image_size,3))
for layer in resNet152.layers:
    layer.trainable = False
model = Sequential()
model.add(resNet152)
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
w_path='C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/logs/'
checkpoint = ModelCheckpoint(w_path+"resNet152.h5",monitor="val_acc",verbose=1,save_best_only=True,
                             save_weights_only=False,period=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
history = model.fit(x_train,y_train,batch_size=32,epochs=50,validation_data=(x_test,y_test)
            ,callbacks=[checkpoint,earlystop])

loss,accuracy = model.evaluate(x_test,y_test)
print("loss:",loss)
print("Accuracy:",accuracy)

y_pred = model.predict_classes(x_test)

show_history(history)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Training_history resNet152.png",
             title="Training history resNet152")

model.save('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/model/'+'resNet152_eyedisease.h5')
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat = cm,figsize=(8,7),class_names = ["Normal","Cataract"],show_normed = True);
plt.savefig('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/resNet152 predict confusion matrix.png')                 


plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]
    if category== 0:
        label = "Normal"
    else:
        label = "Cataract"      
    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "Cataract"   
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout() 
plt.suptitle("Predict show resNet152",fontsize=30)
plt.savefig("C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Predict show resNet152.png")


inceptionResNetV2=InceptionResNetV2(weights="imagenet",include_top = False,input_shape=(image_size,image_size,3))
for layer in inceptionResNetV2.layers:
    layer.trainable = False
model = Sequential()
model.add(inceptionResNetV2)
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
w_path='C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/logs/'
checkpoint = ModelCheckpoint(w_path+"inceptionResNetV2.h5",monitor="val_acc",verbose=1,save_best_only=True,
                             save_weights_only=False,period=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
history = model.fit(x_train,y_train,batch_size=32,epochs=50,validation_data=(x_test,y_test)
            ,callbacks=[checkpoint,earlystop])

loss,accuracy = model.evaluate(x_test,y_test)
print("loss:",loss)
print("Accuracy:",accuracy)

y_pred = model.predict_classes(x_test)

show_history(history)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Training_history inceptionResNetV2.png",
             title="Training history inceptionResNetV2")

model.save('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/model/'+'inceptionResNetV2_eyedisease.h5')
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat = cm,figsize=(8,7),class_names = ["Normal","Cataract"],show_normed = True);
plt.savefig('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/inceptionResNetV2 predict confusion matrix.png')                 


plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]
    if category== 0:
        label = "Normal"
    else:
        label = "Cataract"      
    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "Cataract"   
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout() 
plt.suptitle("Predict show inceptionResNetV2",fontsize=30)
plt.savefig("C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Predict show inceptionResNetV2.png")



xception=Xception(weights="imagenet",include_top = False,input_shape=(image_size,image_size,3))
for layer in xception.layers:
    layer.trainable = False
model = Sequential()
model.add(xception)
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
w_path='C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/logs/'
checkpoint = ModelCheckpoint(w_path+"xception.h5",monitor="val_acc",verbose=1,save_best_only=True,
                             save_weights_only=False,period=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
history = model.fit(x_train,y_train,batch_size=32,epochs=50,validation_data=(x_test,y_test)
            ,callbacks=[checkpoint,earlystop])

loss,accuracy = model.evaluate(x_test,y_test)
print("loss:",loss)
print("Accuracy:",accuracy)

y_pred = model.predict_classes(x_test)

show_history(history)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Training_history xception.png",
             title="Training history xception")

model.save('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/model/'+'xception_eyedisease.h5')
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat = cm,figsize=(8,7),class_names = ["Normal","Cataract"],show_normed = True);
plt.savefig('C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/xception predict confusion matrix.png')                 

plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]
    if category== 0:
        label = "Normal"
    else:
        label = "Cataract"      
    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "Cataract"   
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout() 
plt.suptitle("Predict show xception",fontsize=30)
plt.savefig("C:/Users/GIGABYTE/Downloads/Ocular Disease Recognition/img/Predict show xception.png")










