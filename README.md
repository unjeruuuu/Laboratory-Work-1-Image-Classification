# Laboratory-Work-1-Image-Classification

##[Link for Google Colab] https://colab.research.google.com/drive/1mDaXlxzhuGJizEEzRDv6P8odJ9-5vdz0?usp=sharing

##1. What is the Fashion MNIST dataset?
# - Fashion MNIST is a popular image classification dataset used in machine learning and deep learning as a more challenging replacement for the original MNIST handwritten digits dataset. It contains 70,000 grayscale images of fashion items, each sized 28 × 28 pixels, divided into 60,000 training images and 10,000 test images. The dataset includes 10 clothing categories: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot. Fashion MNIST uses the same format as MNIST, making it easy to integrate into existing models, but it is more realistic and complex, which helps improve model evaluation. It is widely used for learning, benchmarking, and comparing image classification algorithms, especially convolutional neural networks (CNNs), in computer vision and data science.

##2. Why do we normalize image pixel values before training?
# - We normalize image pixel values before training to scale them to a small, consistent range (usually 0 to 1 or −1 to 1), which helps the model learn more efficiently. Normalization prevents large pixel values from dominating calculations, improves numerical stability, speeds up convergence during training, and allows gradient-based algorithms to work more effectively. It also helps ensure that different features contribute equally to the learning process, leading to better performance and more stable training.

##3. List the layers used in the neural network and their functions.
# - Here are the common layers used in a neural network for image classification and their functions:
# - Input Layer – Receives the image data (pixel values) as input to the network.
# - Flatten Layer – Converts the 2D image matrix into a 1D vector so it can be processed by dense layers.
# - Dense (Fully Connected) Layer – Learns patterns and relationships in the data by applying weights and biases to inputs.
# - Activation Layer (e.g., ReLU) – Introduces non-linearity, allowing the network to learn complex patterns.
# - Output Layer (Dense + Softmax) – Produces the final classification probabilities for each class, with Softmax ensuring they sum to 1.

##4. What does an epoch mean in model training?
# - An epoch in model training refers to one complete pass of the entire training dataset through the neural network. During an epoch, the model sees every training sample once and updates its weights based on the loss and gradients. Training usually involves multiple epochs, allowing the model to gradually learn patterns, reduce errors, and improve accuracy. Too few epochs may cause underfitting, while too many can lead to overfitting.

##5. Compare the predicted label and actual label for the first test image.
# - Actual label → the true class of the first test image.
# - Predicted label → the class the model thinks the image belongs to.

##6. What could be done to improve the model’s accuracy?
# - To improve a model’s accuracy, you can increase the training data or use data augmentation, normalize and clean inputs, and make the model more complex with additional layers or neurons. Using effective activation functions, applying regularization like dropout, and tuning hyperparameters such as learning rate and batch size can also help. Leveraging pretrained models through transfer learning and using techniques like early stopping further improve performance and prevent overfitting, leading to better accuracy on both training and test sets.
