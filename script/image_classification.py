# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model

model = load_model("model1.h5")
# summarize model.
model.summary()
