"""
https://keras.io/guides/transfer_learning/
The typical transfer-learning workflow
This leads us to how a typical transfer learning workflow can be implemented in Keras:

Instantiate a base model and load pre-trained weights into it.
Freeze all layers in the base model by setting trainable = False.
Create a new model on top of the output of one (or several) layers from the base model.
Train your new model on your new dataset.
Note that an alternative, more lightweight workflow could also be:

Instantiate a base model and load pre-trained weights into it.
Run your new dataset through it and record the output of one (or several) layers from the base model. This is called feature extraction.
Use that output as input data for a new, smaller model.

Here we use the double parentheses syntax (example, transfer learning):
    model = VGG19(weights='imagenet',include_top=False)
    model.trainable=False
    layer1 = Flatten(name='flat')(model)
    layer2 = Dense(512, activation='relu', name='fc1')(layer1)
    layer3 = Dense(512, activation='relu', name='fc2')(layer2)
    layer4 = Dense(10, activation='softmax', name='predictions')(layer3)

"""
