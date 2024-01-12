# CarSegmentor-using-transfer-learning

**Description**

The "Car Segmentor using Transfer Learning" project leverages advanced computer vision techniques to tackle the challenge of accurately segmenting cars within images. At the heart of this endeavor is the application of transfer learning, a powerful methodology that enhances model performance by leveraging knowledge from a pre-trained model.

The project utilizes VGG16, a state-of-the-art pre-trained convolutional neural network (CNN), as the encoder for a UNet architecture. The incorporation of VGG16 as the encoder enhances the model's ability to generalize and extract meaningful features from images. This is particularly crucial when working with a dataset that is highly specific, presenting a unique challenge for the model's adaptability.

**Dataset**

The Carvana dataset is characterized by its specificity, presenting a unique challenge for models due to its focused nature. Each car in the dataset is meticulously represented with a substantial collection of 16 images, each captured from distinct angles. The controlled studio environment ensures a consistent background across all these images.
[For more informations.](https://www.kaggle.com/c/carvana-image-masking-challenge/data)

**Challenge**

The Dice scores calculated on the Carvana test dataset for the first model (Non-pretrained U-Net) and the secondary model (U-Net with a pretrained VGG16 encoder) showcase impressive segmentation performance. Specifically, the primary model achieves a Dice score of 99.74, while the second model achieves a slightly higher score of 99.75.

Beyond the specific conditions of the Carvana dataset, the central challenge of this project lies in scrutinizing the models' capacity for generalization across a spectrum of real-world scenarios. To tackle this challenge, the project broadens its evaluation criteria by incorporating real examples randomly sourced from the internet. The comparison between the two models becomes pivotal in pinpointing the specific juncture where the second model, enriched with a pretrained VGG16 encoder, has been specifically developed to enhance generalization capabilities.

**User Interface**

To evaluate and validate our artificial model, we created an user interface using the Python library Streamlit, designed for local development and prototyping of the project.
