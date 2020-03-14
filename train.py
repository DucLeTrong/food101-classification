from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Dropout, Activation, Flatten,AveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K

from utils import *
    
def schedule(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return .0002
    elif epoch < 15:
        return 0.00002
    else:
        return .0000005


shape = (224, 224, 3)
X_train, X_test = setup_generator('food-101/train', 'food-101/test', 32, shape[:2])

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
x = base_model.output
x = AveragePooling2D()(x)

x = Dropout(.5)(x)
x = Flatten()(x)
predictions = Dense(X_train.num_classes, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

opt = SGD(lr=.1, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('model.log')


lr_scheduler = LearningRateScheduler(schedule)
model.summary()

model.fit_generator(X_train, validation_data=X_test,
                              epochs=50,
                              steps_per_epoch=X_train.samples//32,
                              validation_steps=X_test.samples//32,
                               callbacks=[lr_scheduler, csv_logger, checkpointer])