from re import I
import librosa,sys
import matplotlib.pyplot as plt
import numpy as np
from librosa import display
import scipy
import math
import os
from random import *
from skimage import io
from skimage import color
from skimage.util import img_as_ubyte
import os
import tensorflow as tf
import keras.layers
segment_length=44100
shift=44100
sampling_rate=44100

def fft_plot(audio, sampling_rate):
  n=len(audio)
  T= 1/sampling_rate
  yf= scipy.fft.fft(audio)
  xf = np.linspace(0.0, 1.0/(2.0*T), n//2)
  fig, ax = plt.subplots()
  ax.plot(xf, 2.0/n * np.abs(yf[:n//2]))
  plt.grid()
  plt.xlabel("Frequency -->")
  plt.ylabel("Magnitude")
  return plt.show()

def spectrogram(samples, sample_rate, stride_ms = 10.0, window_ms = 20.0, min_freq=2000, max_freq = 10000, eps = 1e-14):
 
  stride_size = int(0.001 * sample_rate * stride_ms)
  window_size = int(0.001 * sample_rate * window_ms)

  # Extract strided windows
  truncate_size = (len(samples) - window_size) % stride_size
  samples = samples[:len(samples) - truncate_size]
  nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
  nstrides = (samples.strides[0], samples.strides[0] * stride_size)
  windows = np.lib.stride_tricks.as_strided(samples, 
                                        shape = nshape, strides = nstrides)
    
  assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

  # Window weighting, squared Fast Fourier Transform (fft), scaling
  weighting = np.hanning(window_size)[:, None]
    
  fft = np.fft.rfft(windows * weighting, axis=0)
  fft = np.absolute(fft)
  fft = fft**2
    
  scale = np.sum(weighting**2) * sample_rate
  fft[1:-1, :] *= (2.0 / scale)
  fft[(0, -1), :] /= scale
    
  # Prepare fft frequency list
  freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
  
  # Compute spectrogram feature
  startind=np.where( freqs >=min_freq)[0][0] 
  endind = np.where( freqs <=max_freq)[0][-1] + 1
  #print(str(startind)+":"+str(endind))
  specgram = np.log(fft[startind:endind, :] + eps)
  return specgram
  
def plot_bird(file_path):
  samples, sampling_rate = librosa.load(file_path, sr= None, mono=True, offset=0.0, duration = None)
  spec=spectrogram(samples,sampling_rate)
  plt.imshow(spec)
  plt.imsave("file_path")
  plt.xlabel(file_path+" Time frame windows")
  plt.ylabel("Frequency")
  plt.show()

def eval_amplitude(samples):
  max=-1
  sum=0.0
  for s in samples:
    #print(s)
    amp=s*s
    if amp > max:
      max=amp
    sum=sum+amp
  return(max,math.sqrt(sum/len(samples)),math.sqrt(sum))

def splice_sound_file(file_path,splice_length,shift):
  samples, sampling_rate = librosa.load(file_path, sr= None, mono=True, offset=0.0, duration = None)
  print("file length ",len(samples))
  sounds=[]
  remaining=len(samples)
  start=0
  ii=0
  #print("splice_length "+str(splice_length))
  #print("shift "+str(shift))
  while(remaining > 0 ):
    #print("remaining: "+str(int(remaining)))
    end=splice_length
    if(remaining < splice_length):
      if(remaining > (splice_length/2)):
        #more than half of time remaining
        #back it up
        #print("greater than half") 
        start -=int((splice_length-remaining)) 
        end=int(splice_length)        
        if start < 0:
          #print("start less than 0")
          #clip is less than splice_length, add empty to end
          start=0
          end=remaining
          sound=samples[start:end]
          add=np.empty((1,splice_length-end))
          add[:]=0.0
          sound=np.append(sound,add)
          sounds.append(sound)
          return sounds
      else:
        return sounds
    #print(str(int(start))+":"+str(int(start+end)))
    sounds.append(samples[int(start):int(start+end)])
    remaining =remaining -shift
    start=start+shift
    ii+=1
  return sounds

def spec_to_file(spec, img_file_path):
  #plt.axis('off')
  #plt.axis('tight')
  np.save(img_file_path+".npy",spec)
  img=plt.imshow(spec)
  plt.savefig(img_file_path+".png")
  #now save data in .txt file
  
#Note: This will parse all subdirectories of path.
def process_training_dir_audio(path,specpath,label,labels,sampling_rate,segment_length, shift, minAmp=0.01):
  print(label)
  images=[] #np.empty((0,161,149))
  test_images=[] #np.empty((0,161,149))
  classes=[]
  test_classes=[]
  display=False
  dirs = os.listdir(path)
  useLabel=True
  dirCount=0
  for entry in dirs:
    file_path=os.path.join(path, entry)
    print(file_path)
    if os.path.isfile(file_path):
      if  ".wav" in file_path:
        
        if useLabel:
          labels.append(label);
          useLabel=False;

          print("label "+str(label))
          #rand=1.0 #always put first of each class to test
        classIndex=int(len(labels)-1)
        testFile=False
        #print("processing file: "+file_path)
        if file_path.find("test") >=0:
          testFile=True

        sounds=splice_sound_file(file_path,segment_length,shift)
        countAmps=0
        #print("Num sounds "+str(len(sounds)))
        for sound in sounds:
          (maxAmp,avgAmp,sumAmp)=eval_amplitude(sound)
          print("Avg amp: "+str(avgAmp)+" sum Amp: "+str(sumAmp))
          if maxAmp >= minAmp:
            #Only use if file reaches minimum amplitude
            countAmps+=1
            dirCount=dirCount+1
            spec=spectrogram(sound,sampling_rate)
            img_file=os.path.join(specpath,label+"_"+str(dirCount))
            if(testFile):
              print("test file "+file_path)
              img_file=img_file+"_test"
              test_images.append(spec)
              test_classes=np.append(test_classes,classIndex)
              #rand=0.1 #reset rand to lower
            else:
              images.append(spec)
              #class is the index of the current label
              classes=np.append(classes, classIndex)
            if(display):
              plt.imshow(spec)
              plt.xlabel(path+" Time frame windows") 
              plt.ylabel("Frequency")
              plt.show()
            spec_to_file(spec,img_file)
            #print(img_file)
            # print(spec.dtype)
            # img=(color.convert_colorspace(spec, 'HSV', 'RGB')*255).astype(np.uint8)
            # print(spec.dtype)
            #skimage.io.imsave(img_file, spec)
            #plt.savefig(img_file)
          
            
        #print("sounds over "+str(minAmp)+" amps: "+str(countAmps)+" out of "+ str(len(sounds)))
    else:
      #if its a dir, then the dir's name is the label
      dirCount=0
      print("Dir: "+entry)
      (images2,classes2,test_images2,test_classes2)=process_training_dir_audio(os.path.join(path, entry),os.path.join(specpath, entry),entry,labels,sampling_rate,segment_length,shift,minAmp)
      #print("Num sounds: "+str(len(images2))+" tests sounds: "+str(len(test_images2)))
      if len(images) == 0:
        images=images2
      else:
        print("images: "+str(len(images)))
        print("images2: "+str(len(images2)))
        images = np.append(images,images2,axis=0)
      if len(test_images) == 0:
        test_images = test_images2
      elif len(test_images2) > 0:
        test_images=np.append(test_images,test_images2,axis=0)
      classes=np.append(classes,[classes2])
      test_classes=np.append(test_classes,test_classes2)
  print("finished directory")
  if(len(test_images) > 0):
    test_images=np.stack(test_images)
    #print(test_images.shape)
  if(len(images) > 0):
    images=np.stack(images)
    #print(images.shape)
  return (images,classes,test_images,test_classes)



def process_dir_audio(path,specpath,label,labels,sampling_rate,segment_length, shift, minAmp=0.01):
  images=[] #np.empty((0,161,149))
  display=False
  dirs = os.listdir(path)
  useLabel=True
  dirCount=0
  for entry in dirs:
    file_path=os.path.join(path, entry)
    print("current file: "+file_path)
    if os.path.isfile(file_path):
      if  ".wav" in file_path or ".WAV" in file_path:
        
        if useLabel:
          labels.append(label);
          useLabel=False;

          print("label "+str(label))

        sounds=splice_sound_file(file_path,segment_length,shift)
        countAmps=0
        print("Num sounds "+str(len(sounds)))
        for sound in sounds:
          (maxAmp,avgAmp,sumAmp)=eval_amplitude(sound)
          #print("Max amp: "+str(maxAmp)+" avg Amp: "+str(avgAmp))
          if maxAmp >= minAmp:
            #Only use if file reaches minimum amplitude
            countAmps+=1
            dirCount=dirCount+1
            spec=spectrogram(sound,sampling_rate)
            img_file=os.path.join(specpath,label+"_"+str(dirCount))
            images.append(spec)
            #class is the index of the current label
            if(display):
              plt.imshow(spec)
              plt.xlabel(path+" Time frame windows") 
              plt.ylabel("Frequency")
              plt.show()
            spec_to_file(spec,img_file)
            # print(spec.dtype)
            # img=(color.convert_colorspace(spec, 'HSV', 'RGB')*255).astype(np.uint8)
            # print(spec.dtype)
            #skimage.io.imsave(img_file, spec)
            #plt.savefig(img_file)
          
            
        #print("sounds over "+str(minAmp)+" amps: "+str(countAmps)+" out of "+ str(len(sounds)))
    else:
      #if its a dir, then the dir's name is the label
      dirCount=0
      images2=process_dir_audio(os.path.join(path, entry),os.path.join(specpath, entry),entry,labels,sampling_rate,segment_length,shift,minAmp)
      #print("Num sounds: "+str(len(images2))+" tests sounds: "+str(len(test_images2)))
      if len(images) == 0:
        images=images2
      else:
       
        images = np.append(images,images2,axis=0)
      
  if(len(images) > 0):
    images=np.stack(images)
    print(images.shape)
  return images

def spec_from_file(spec_file_path):
  spec = np.load(spec_file_path)
  return spec

def process_training_dir_spectrograms(path,label,labels):
  print(label)
  images=[] #np.empty((0,161,149))
  test_images=[] #np.empty((0,161,149))
  classes=[]
  test_classes=[]
  display=False
  dirs = os.listdir(path)
  useLabel=True
  dirCount=0
  testPercent=0.05
  for entry in dirs:
    file_path=os.path.join(path, entry)
    if os.path.isfile(file_path):
      if  ".npy" in file_path:
        if useLabel:
          labels.append(label);
          useLabel=False;
          print("label "+str(label))
          #rand=1.0 #always put first of each class to test
        classIndex=int(len(labels)-1)
        testFile=False
        print("processing file: "+file_path)
        spec=spec_from_file(file_path)
        #if file_path.find("test") >=0:
        #  testFile=True
        if random() < testPercent:
          testFile=True
        if(testFile):
          print("test file "+file_path)
          test_images.append(spec)
          test_classes=np.append(test_classes,classIndex)
          #rand=0.1 #reset rand to lower
        else:
          images.append(spec)
          #class is the index of the current label
          #print(file_path+" "+str(classIndex))
          classes=np.append(classes, classIndex)
            
        #print("sounds over "+str(minAmp)+" amps: "+str(countAmps)+" out of "+ str(len(sounds)))
    else:
      #if its a dir, then the dir's name is the label
      dirCount=0
      (images2,classes2,test_images2,test_classes2)=process_training_dir_spectrograms(os.path.join(path, entry),entry,labels)
      if len(images) == 0:
        images=images2
      else:

        images = np.append(images,images2,axis=0)
      if len(test_images) == 0:
        test_images = test_images2
      elif len(test_images2) > 0:
        test_images=np.append(test_images,test_images2,axis=0)
      classes=np.append(classes,[classes2])
      test_classes=np.append(test_classes,test_classes2)
 
  if(len(test_images) > 0):
    test_images=np.stack(test_images)
    #print(test_images.shape)
  if(len(images) > 0):
    images=np.stack(images)
    print(images.shape)
  return (images,classes,test_images,test_classes)

def create_normal_model(images, classes, test_images,test_classes, labels,debug=False):
  if debug:
    print(images.shape)
    print(test_images.shape)
    print(str(len(classes))+","+str(len(test_classes)))

  layer = keras.layers.Normalization()
  layer.adapt(images) 
  normalized_images = layer(images)
  print(normalized_images[0])

  #print("Normalization layer: ")
  #print(layer)
  #use same adaptation as train images for the others
  layer = keras.layers.Normalization()
  layer.adapt(test_images) 
  normalized_test_images=layer(test_images)
  model=tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=normalized_images[0].shape),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(len(labels),activation='softmax')
      ])


  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

  model.fit(normalized_images, classes, epochs=10)


  if len(test_images > 0):
      test_loss, test_acc = model.evaluate(normalized_test_images,  test_classes, verbose=3)
      print('\nTest accuracy:', test_acc)
  return model
      

  # Helper libraries
  import numpy as np
  import matplotlib.pyplot as plt

def create_cnn_model(images, classes, test_images,test_classes, labels,debug=False):
  ### CNN version
  if debug:
        print(images.shape)
        print(test_images.shape)
        print(str(len(classes))+","+str(len(test_classes)))
  model=tf.keras.Sequential(
      [
          tf.keras.Input(shape=(161,99,1)),
          tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
          tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
          tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(len(labels))
          
      ]
  )

  print(model.summary())
  # model=tf.keras.Sequential()
  # model.add(layers.Conv2D(161, (3, 3), activation='relu', input_shape=images[0].shape))
  # model.add(layers.MaxPooling2D((2, 2)))
  # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  # model.add(layers.MaxPooling2D((2, 2)))
  # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  # model.add(layers.Flatten())
  # model.add(layers.Dense(64, activation='relu'))
  # model.add(layers.Dense(len(labels)))

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  if len(test_images > 0):
      history = model.fit(images, classes, epochs=5, validation_data=(test_images, test_classes))
  else:
      history = model.fit(images, classes, epochs=5)

  plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0.5, 1])
  plt.legend(loc='lower right')
  return model

  def print_images(images,labels,classes):
        # i=0
    # for img in images:
    #   print(labels[int(classes[i])])
    #   i=i+1
    #   plt.imshow(img)
    #   plt.show()

    print("*****test images******")

    i=0
    for img in images:
      print(labels[int(classes[i])])
      
      i=i+1
      
      
      
      plt.imshow(img)
      plt.show()

  import math
def plot_image(i, predictions_array, labels,true_label, img):
  true_label, img = true_label[i], img[i]
  #plt.subplots_adjust(bottom=0.25, right=0.25, top=0.25)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(labels[predicted_label],
                                100*np.max(predictions_array),
                                labels[true_label]+str(i)),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(len(predictions_array)))
  plt.yticks([])
  thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def make_predictions(model, images):                                                  
  #model.summary()  
  probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  predictions = probability_model.predict(images)
  return (probability_model,predictions)


def load_images_from_audio():
  labels=[]
  file_path= os.path.join("D:",os.sep,"Grassland","Data","Recordings","Freedom")
  spec_path= os.path.join("D:",os.sep,"Grassland","Data","Spectrograms","Freedom")
  (images)=process_dir_audio(file_path,spec_path,"",labels,sampling_rate,segment_length)
  return images

def process_locations(model,classLabels,path,sampling_rate,segment_length,minAmp,buckets={}):
  dirs = os.listdir(path)
  currLoc=0
  #two layers of dirs
  for entry in dirs:
    currLoc=currLoc+1
    loc_path=os.path.join(path, entry)
    if os.path.isdir(loc_path):
      if not entry in buckets:
        #no bucket set for this location yet
        buckets[entry]=[0 for i in range(len(classLabels))] 
      loc_bucket=buckets[entry]
      #print("processing "+loc_path)
      files=os.listdir(loc_path)
      for file in files:
        process_location_file(model,os.path.join(loc_path,file),buckets[entry],sampling_rate,segment_length,segment_length,minAmp)
  for b in buckets:
    bucket=buckets[b]
    for i in range(len(bucket)):
      print("class "+str(i)+" total: "+str(round(bucket[i],2)))
    
def process_location_file(model,path,buckets,sampling_rate=44100,segment_length=44100,shift=44100,minAmp=0.01):
  #process a file from a location and add the amplitude to the bucket for each identified class
  display=False
  images=[] #np.empty((0,161,149))
  
  classes=[]
  amps=[]
  divisor=0.01
  save_path=os.path.join("D:",os.sep,"Grassland","Data","Classified","class_")
  if  (".wav" in path or ".WAV" in path) and filter in path:
    print("processing "+path)
    sounds=splice_sound_file(path,segment_length,shift)
    count=0
    for sound in sounds:
      count = count+1
      (maxAmp,avgAmp,sumAmp)=eval_amplitude(sound)
      if display:
        print("Sum amp: "+str(sumAmp)+" avg Amp: "+str(avgAmp))
      if maxAmp >= minAmp:
        #Only use if file reaches minimum amplitude
        spec=spectrogram(sound,sampling_rate)
        images.append(spec)
        amps.append(sumAmp)
        # if(display):
        #   plt.imshow(spec)
        #   plt.xlabel(path+" Time frame windows")
        #   plt.ylabel("Frequency")
        #   plt.show()
        #now find class of image
    if len(images) >0:
      images=np.stack(images)
      #normalize first
      layer = keras.layers.Normalization()
      layer.adapt(images) 
      normalized_images = layer(images)
      
      print("about to make predictions on "+str(len(images))+" images.")
      (probability_model,predictions)=make_predictions(model,normalized_images)
      numImages=len(normalized_images)
      display2=False
      for i in range(numImages):
        display2=False
        #print(str(i)+" of "+str(numImages))
        predict=predictions[i]
        amp=amps[i]
        cls=np.argmax(predict)
        print("most likely class: "+str(cls) + " "+str(amp))
        if cls==0:
          display2=True
        if display or display2:
          plt.imshow(images[i])
          plt.show() 
          spec_to_file(images[i],save_path+str(cls)+"_"+str(i))
        for c in range(len(predict)):
          #add the probabilistic amplitude to bucket (amp * probability of class c)
          buckets[c]+=amp*predict[c]*divisor
          if (display or display2):
            #print("predicted class: "+str(c))
            print("class: "+str(c)+" likelihood: "+str(round(predict[c],3))+" total value: "+str(round(amp*predict[c])))

  #process location sound files with a saved model

import csv

def load_bucket_file(file_path, num_classes):
  buckets={}
  #encode as follows to remove first BOM character 
  with open(file_path,encoding='utf-8-sig') as csvDataFile:
      csvReader = csv.reader(csvDataFile)
      for row in csvReader:
        location=row[0]
        classNum=int(row[1])
        value=float(row[2])
        if not location in buckets:
          b = [0 for i in range(num_classes)] 
          buckets[location]=b
        bucket=buckets[location]
        bucket[classNum]=value
  return buckets
          
def save_bucket_file(file_path,buckets):
  with open(file_path, 'w', newline='',encoding='utf-8-sig') as csvfile:
    csvWriter=csv.writer(csvfile,delimiter=',')
    
    for loc in buckets.keys():
      bucket=buckets[loc]
      i=0
      for v in bucket: #each class
        row=[loc,str(i),str(v)]
        i=i+1 #increment class num
        csvWriter.writerow(row)


#if make spectrograms is true, then images need to be made from audio files.
#otherwise load training from image files
def build_models_from_training(spec_path=None, file_path=None,make_spectrograms=False):
  if make_spectrograms: 
      
      print(file_path)
      (images,classes,test_images,test_classes)=process_training_dir_audio(file_path,spec_path,"Birdsongs",labels,sampling_rate,segment_length,shift)
  else: #audio has already been saved to spectrograms
      labels=[]#=["background_noise","bobof_whine","bobom_buzz","bobom_song","bobo_chunk","bobo_flight","chipping_sparrow","goldfinch","hammer","house_sparrow","meadowm","meadow_chatter","red_eyed_vireo","robin","savannah_sparrow","saw","song_sparrow","tree_swallow","warblers","wind"]
      (images,classes,test_images,test_classes)=process_training_dir_spectrograms(spec_path,"BirdSongs",labels)
  images=np.asarray(images)
  print(images.shape)
  classes=classes.astype(int)
  test_classes=test_classes.astype(int)
  model=create_cnn_model(images,classes,test_images,test_classes,labels)
  return model
                    

def analyze_recordings(bucket_file,labels,file_path,model,filter="",):
  buckets=load_bucket_file(bucket_file,len(labels))
  print(buckets)
  process_locations(model,labels,file_path,44100,44100,0.01,buckets)
  save_bucket_file(bucket_file,buckets)

#    model.save(model_path)

if __name__ == '__main__':
      path=os.path.join("C:",os.sep,"Users","greeneks","OneDrive - Thomas College","Documents-PC","Birds","Grassland")
      spec_path= os.path.join(path,os.sep,"Data","Training","Spectrograms")
      file_path= os.path.join(path,os.sep,"Data","Training","Audio")
      bucket_file= os.path.join(path,os.sep,"Analysis","Richardson Analysis","bird_buckets_6_1.csv")
      data_path=os.path.join(path,os.sep,"Data","Recordings","Richardson 2024")
      labels=["bobof_whine","bobom","misc"]
      model_path=os.path.join(path,"Models","latest_model")
      build_model=True
      if build_model:
            model=build_models_from_training(spec_path,file_path,True)
            model.save(model_path)
      else:
            model=tf.keras.models.load_model(model_path)
            
      filter="0601"   #to analyze only a specific date filter
      analyze_recordings(bucket_file,labels,file_path,model,filter)
      #C:\Users\greeneks\OneDrive - Thomas College\Documents-PC\Birds
