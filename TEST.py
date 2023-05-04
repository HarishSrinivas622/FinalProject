
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('mee.h5')

img=image.load_img('Datasets\\test\\MildDemented\\MildDem (148).jpg',target_size=(224,224))


x=image.img_to_array(img)
x

x.shape

x=x/255


import numpy as np
x=np.expand_dims(x,axis=0)
img_data=(x)
img_data.shape

model.predict(img_data)

print("finAL")
print(model.predict(img_data))

a=np.argmax(model.predict(img_data), axis=1)

print(a)



# Step 1
score_theory = 60
score_practical = 20

if(a == 0):
    print("The given image is normal") # type 1
elif(a ==1):
    print("The given image is Astrocitoma")  # Type 2
elif(a ==2):
    print("Ependimoma")  # Type 2


else:
    print("Not applicable ") # not applicable

    
