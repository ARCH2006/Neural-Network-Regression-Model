# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network model that predicts a continuous numerical output based on a single input feature. The model consists of three fully connected layers with ReLU activation functions and is optimized using the RMSprop optimizer with Mean Squared Error (MSE) loss.

## Neural Network Model

![Screenshot 2025-03-21 135447](https://github.com/user-attachments/assets/d1e899a5-04e9-4bbd-bff0-a0b9f099ba86)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: ARCHANA S
### Register Number: 212223040019
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
                self.fc2 = nn.Linear(8,10)
                self.fc3 = nn.Linear(10,1)
                #activation func
                self.relu = nn.ReLU()
                self.history = {'loss' : []}
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) #no activation func in O/P
        return x

      



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet() #ai_brain is a model
criterion = nn.MSELoss() #loss function
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001) #to adjust the weights



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs): 
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward() #adjusting according to loss
        optimizer.step()
    
        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information
![image](https://github.com/user-attachments/assets/9a6dcf59-b4a8-4fb8-9c65-086de0f6c96d)



## OUTPUT
![image](https://github.com/user-attachments/assets/a5a977fc-198d-4e6c-91b0-51bc9e3ccd95)


### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/4284c111-6ec6-4169-a3a0-0aeb2c09721f)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/a5051ae4-33ad-48d7-96a5-51fa64a0661e)


## RESULT
Thus, the neural network regression model for the given dataset is successfully developed.
