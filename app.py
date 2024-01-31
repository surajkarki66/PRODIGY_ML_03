import pickle

from utils import process_input


# Load the pre-trained linear regression model
with open('catvsdog.pkl', 'rb') as file:
    MODEL = pickle.load(file)

if __name__ == '__main__':
    
    path = "./d.png"
    img = process_input(path)
    pred = MODEL.predict(img)[0]
    print(pred)
    if pred == 0:
        print("The given image is of Cat!")
    else:
        print("The given image is of Dog!")