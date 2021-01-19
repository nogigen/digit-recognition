from tkinter import *
import PIL
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import pickle
import os

class MLP(object):

    def __init__(self, num_inputs, num_hidden_layers, num_outputs, weights, biases):
        self.layers = [num_inputs] + num_hidden_layers + [num_outputs]
        self.weights = weights
        self.biases = biases

        self.activations = []
        # create activations per layer
        for i in range(len(self.layers)):
            activation = np.zeros(self.layers[i])
            self.activations.append(activation)
            


    
    def forward(self, inputs):
        self.activations[0] = inputs
        activations = inputs

        for i, weights in enumerate(self.weights):
            z = np.dot(activations, weights) + self.biases[i]

            if i != len(self.weights) - 1:                
                activations = self.relu(z)
            else:
                activations = self.softmax(z)

            self.activations[i + 1] = activations

        return activations # output

    def relu(self, x):
        return x * (x > 0)

    def softmax(self, x):
        values = np.exp(x - np.amax(x))
        values = values / np.sum(values)
        values_clipped = np.clip(values, 1e-7, 1 - 1e-7)
        return values_clipped



def predict():

    filename = "image.png"
    image.save(filename)

    img = Image.open(filename).convert("L")
    
    new_height = 28
    new_width = int(new_height / img.height * img.width)
    img = img.resize((new_height,new_width))
    
    arr = np.array(img)
    os.remove(filename)

    # normalize the array
    newMin, newMax, oldMin, oldMax = (0 , 1 , arr.min() , arr.max())

    arr = (arr - oldMin) * ((newMax - newMin) / (oldMax - oldMin)) + newMin
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] == 0:
                arr[i][j] = 1
            else:
                arr[i][j] = 0
    
    output = mlp.forward(arr.flatten())
    prediction = np.argmax(output)
    confidenceLevel = output[prediction]

    labelClass.set("Labeled as : {}" .format(prediction))
    labelConfidence.set("Confidence level : {:.2f}% ".format(confidenceLevel * 100))

    img.save(filename)


def paint(event):
    color = 'black'
    x1, y1 = (event.x -1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1,y1,x2,y2, fill=color, outline=color)
    draw.line([x1,y1,x2,y2], fill="black", width=8)

def reset():
    global image,draw
    canvas.delete("all")
    labelClass.set("Labeled as : ")
    labelConfidence.set("Confidence level : ")

    image = PIL.Image.new("RGB", (canvas_width, canvas_height), (255,255,255))
    draw = ImageDraw.Draw(image)

    for filename in os.listdir():
        if filename == "image.png":
            os.remove(filename)
            break



# while resizing the imgs to 28x28, img's quality gets really bad. Normally this NN has 97% accuracy in the MNIST database


# load num_inputs, num_hidden_layers, num_outputs, weights, biases from txt files
pickle_in = open("weights.pickle", "rb")
weights = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("biases.pickle", "rb")
biases = pickle.load(pickle_in)
pickle_in.close()

file = open("layers.txt", "r")
line = file.read().strip().split(" ")
file.close()

num_inputs = int(line[0])
num_outputs = int(line[len(line) - 1])

line.pop(0)
line.pop(len(line) - 1)

num_hidden = [int(num) for num in line]

mlp = MLP(num_inputs, num_hidden, num_outputs, weights, biases)

root = Tk()

root.geometry("+700+350")
root.geometry("450x390")
root.resizable(0,0)

topLabel = Label(root, text="Digit Recognizer")
topLabel.pack()

canvas_width = 280
canvas_height = 280
canvas = Canvas(root, width= canvas_width, height= canvas_height, bg="white")
canvas.pack()
canvas.bind('<B1-Motion>', paint)

image = PIL.Image.new("RGB", (canvas_width, canvas_height), (255,255,255))
draw = ImageDraw.Draw(image)

bottomFrame = LabelFrame(root, borderwidth=0, pady=10)
bottomFrame.pack()

buttonPredict = Button(bottomFrame, text="predict", command=predict).grid(row=0,column=0)

buttonReset = Button(bottomFrame, text="reset", command=reset).grid(row=0,column=1)

labelClass = StringVar()
labelClass.set("Labeled as : ")
label2 = Label(bottomFrame, textvariable=labelClass).grid(row=1, column=0, columnspan=2)

labelConfidence = StringVar()
labelConfidence.set("Confidence level : ")
label3 = Label(bottomFrame, textvariable=labelConfidence).grid(row=2, column=0, columnspan=2)


root.mainloop()