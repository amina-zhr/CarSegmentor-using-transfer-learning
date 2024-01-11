import streamlit as st

# Displaying the Table of Contents in the sidebar
st.sidebar.title("Table of Contents")
st.sidebar.markdown("- [Convolutional Neural Networks (CNN)](#cnn)")
st.sidebar.markdown("- [VGG16 Architecture](#vgg16-architecture)")
st.sidebar.markdown("- [U-Net Architecture](#u-net-architecture)")
st.sidebar.markdown("- [Transfer Learning](#transfer-learning)")

# Adding content sections
st.title("Streamlit App with Table of Contents")
st.markdown("""
We will discuss below the most importants notions that we have used in our project :
- Convolutional Neural Networks (CNN)
- VGG16 Architecture
- U-Net Architecture
- Transfer Learning
""")

# Section 1
st.markdown('<a id="cnn"></a>', unsafe_allow_html=True)
st.header("Convolutional Neural Networks (CNN)")
st.markdown("""Upon reflection, it's strange to use networks with fully-connected layers to classify images. The reason is that such a network architecture does not take into account the spatial structure of the images. For instance, it treats input pixels which are far apart and close together on exactly the same footing. Such concepts of spatial structure must instead be inferred from the training data. But what if, instead of starting with a network architecture which is tabula rasa, we used an architecture which tries to take advantage of the spatial structure? A possible answer is <b>convolutional neural networks</b>. These networks use a special architecture which is particularly well-adapted to classify images. Using this architecture makes convolutional networks fast to train. This, in turn, helps us train deep, many-layer networks, which are very good at classifying images.<br><br>Convolutional neural networks use three basic ideas: <b>local receptive fields</b>, <b>shared weights</b>, and <b>pooling</b>. Let's look at each of these ideas in turn.
""", unsafe_allow_html=True)
st.markdown("""
<b>Local receptive fields</b>: In the following we consider a 28 × 28 image which have one channel e.g. grayscale image as colored image need at least three channel(28 × 28 × 3) :
""", unsafe_allow_html=True)
st.image("Streamlit/images/img.png" )
st.markdown("""
As per usual, we'll connect the input pixels to a layer of hidden neurons. But we won't connect every input pixel to every hidden neuron. Instead, we only make connections in small, localized regions of the input image.
<br><br>
To be more precise, each neuron in the first hidden layer will be connected to a small region of the input neurons, say, for example, a 5 × 5
 region, corresponding to 25
 input pixels. So, for a particular hidden neuron, we might have connections that look like this:
""", unsafe_allow_html=True)
st.image("Streamlit/images/img_1.png")
st.markdown("""
That region in the input image is called the local receptive field for the hidden neuron. It's a little window on the input pixels. Each connection learns a weight. And the hidden neuron learns an overall bias as well. You can think of that particular hidden neuron as learning to analyze its particular local receptive field.<br><br>
We then slide the local receptive field across the entire input image. For each local receptive field, there is a different hidden neuron in the first hidden layer. To illustrate this concretely, let's start with a local receptive field in the top-left corner:
""", unsafe_allow_html=True)
st.image("Streamlit/images/img_2.png")
st.markdown("""
Then we slide the local receptive field over by one pixel to the right (i.e., by one neuron), to connect to a second hidden neuron:""", unsafe_allow_html=True)
st.image("Streamlit/images/img_3.png")
st.markdown("""
And so on, building up the first hidden layer. Note that if we have a 28 × 28
 input image, and 5 × 5
 local receptive fields, then there will be 24 × 24
 neurons in the hidden layer. This is because we can only move the local receptive field 23
 neurons across (or 23
 neurons down), before colliding with the right-hand side (or bottom) of the input image.
<br><br>
I've shown the local receptive field being moved by one pixel at a time. In fact, sometimes a different <b>stride length</b> is used. For instance, we might move the local receptive field 2
 pixels to the right (or down), in which case we'd say a stride length of 2
 is used.
""", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<b>Shared weights and biases:</b> I've said that each hidden neuron has a bias and 5 × 5
 weights connected to its local receptive field. What I did not yet mention is that we're going to use the same weights and bias for each of the 24 × 24
 hidden neurons. In other words, for the j,k
th hidden neuron, the output is:
""", unsafe_allow_html=True)
st.latex(r"""
  \sigma\left(b + \sum_{l=0}^4 \sum_{m=0}^4  w_{l,m} a_{j+l, k+m} \right).
""")

st.markdown("""
Here, σ
 is the neural activation function. b
 is the shared value for the bias. wl,m
 is a 5 × 5
 array of shared weights. And, finally, we use ax,y
 to denote the input activation at position x,y
.
""", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
We call the map from the input layer to the hidden layer a <b>feature map</b>. We usually work with several ones, where each map detect a single useful feature. 
""", unsafe_allow_html=True)
st.image("Streamlit/images/img.jpeg", caption="In CNNs we represent the layers as volumes where the depth is the number of channels for example a typical colored image has depth 3. The depth of the convolutional layers represent the number of feature maps.")
st.markdown("---")
st.markdown("""
<b>Pooling layers:</b> In addition to the convolutional layers just described, convolutional neural networks also contain pooling layers. Pooling layers are usually used immediately after convolutional layers. What the pooling layers do is simplify the information in the output from the convolutional layer.
<br><br>
In detail, a pooling layer takes each feature map output from the convolutional layer and prepares a condensed feature map. For instance, each unit in the pooling layer may summarize a region of (say) 2 × 2
 neurons in the previous layer. As a concrete example, one common procedure for pooling is known as <b>max-pooling</b>. In max-pooling, a pooling unit simply outputs the maximum activation in the 2 × 2
 input region, as illustrated in the following diagram:
""", unsafe_allow_html=True)
st.image("Streamlit/images/img_4.png")
# Sectionn 2



st.header("VGG16 Architecture")
st.markdown("""
VGG16 refers to the VGG model, also called VGGNet. It is a convolution neural network (CNN) model supporting 16 layers. <b>K. Simonyan</b> and <b>A. Zisserman</b> from Oxford University proposed this model and published it in a paper called <b>Very Deep Convolutional Networks for Large-Scale Image Recognition</b>.

The VGG16 model can achieve a test accuracy of 92.7% in ImageNet, a dataset containing more than 14 million training images across 1000 object classes. It is one of the top models from the ILSVRC-2014 competition.
""", unsafe_allow_html=True)
st.image("Streamlit/images/img_5.png")
# Section 3
st.header("U-Net Architecture")
st.markdown("""
U-Net is a convolutional neural network that was developed for biomedical image segmentation at the Computer Science Department of the University of Freiburg. The network is based on a fully convolutional neural network whose architecture was modified and extended to work with fewer training images and to yield more precise segmentation. Segmentation of a 512 × 512 image takes less than a second on a modern GPU.

The U-Net architecture has also been employed in diffusion models for iterative image denoising. This technology underlies many modern image generation models, such as <b>DALL-E</b>, <b>Midjourney</b>, and <b>Stable Diffusion</b>.
""", unsafe_allow_html=True)
st.image("Streamlit/images/img_6.png")

st.header("Transfer Learning")
st.markdown("""
Transfer learning (TL) is a technique in machine learning (ML) in which knowledge learned from a task is re-used in order to boost performance on a related task. For example, for image classification, knowledge gained while learning to recognize cars could be applied when trying to recognize trucks.
""", unsafe_allow_html=True)
st.image("Streamlit/images/img_7.png")
st.markdown("---")
st.subheader("What we have used ?")
st.markdown("""
In our case, we have used a pre-trained VGG16 on ImageNet deleting the fully-connected network and 3 last convolutional layers. And Concatenate this to a U-Net architecture that we have deleted the down part from It.

The second part of our model (decoder) is fine-tuned on the Carvana dataset, while the encoder's weights are freezed on the ImageNet dataset.
""",unsafe_allow_html=True)