# **Landmark Detection Project**  

This project focuses on building a **Landmark Detection System** using **Deep Learning**. The goal is to classify images of landmarks by training a neural network model on a dataset of landmark images. The project involves data preprocessing, model building, training, and evaluation.  



## **Objective**  
The primary goal of this project is to develop a **Deep Learning model** capable of identifying and classifying landmarks from images. The model is trained on a dataset containing images of various landmarks, and it learns to predict the correct landmark ID for a given image.  



## **Tools and Technologies Used**  
- **Programming Language**: Python  
- **Libraries**:  
  - Data Processing: `pandas`, `numpy`  
  - Image Processing: `OpenCV`, `PIL`  
  - Visualization: `matplotlib`  
  - Deep Learning: `Keras` (with TensorFlow backend)  
- **Model**:  
  - Sequential Neural Network with Dense layers  



## **Key Features**  
1. **Data Preprocessing**:  
   - Resizing images to a uniform size (128x128).  
   - Normalizing pixel values to the range [0, 1].  
2. **Model Architecture**:  
   - A simple **Sequential Neural Network** with:  
     - Input layer (Flattened image data).  
     - Dense layer with ReLU activation.  
     - Output layer with Softmax activation for multi-class classification.  
3. **Model Training**:  
   - Trained on a subset of the landmark dataset.  
   - Used **Categorical Crossentropy** as the loss function and **Adam** optimizer.  
4. **Evaluation**:  
   - Evaluated the model on a test set to measure accuracy and loss.  


## **How to Run the Code**  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/YEGNESWAR07/Landmark-Detection.git  
   ```  
2. Install the required libraries:  
   ```bash  
   pip install pandas numpy opencv-python pillow matplotlib tensorflow keras  
   ```  
3. Download the dataset:  
   - Place the `train.csv` file and the `images` folder in the project directory.  
4. Run the Python script:  
   ```bash  
   python landmark_detection.py  
   ```  


## **Results**  
- **Training**: The model was trained for 10 epochs, achieving a validation accuracy of **X%** (replace with actual value).  
- **Testing**: The model achieved a test accuracy of **Y%** (replace with actual value).  
- **Visualizations**:  
  - Distribution of landmark IDs in the dataset.  
  - Training and validation loss/accuracy curves.  


## **Future Improvements**  
- Experiment with more complex architectures like **Convolutional Neural Networks (CNNs)** for better performance.  
- Use data augmentation techniques to improve model generalization.  
- Deploy the model as a web application using **Flask** or **Streamlit**.  


## **Contributing**  
Feel free to contribute to this project by:  
- Reporting issues or bugs.  
- Suggesting new features or improvements.  
- Submitting pull requests.  




