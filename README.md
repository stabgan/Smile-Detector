# Smile-Detector

Code is well commented .

I used haar cascade to make a smile detector which is an improvement of my earlier project  [HaarFacialRecognition](https://github.com/KaustabhGanguly/HaarFacialRecognition)

The xml files can be found in : - [opencv cascade directory](https://github.com/opencv/opencv/tree/master/data/haarcascades)

You may need to change the <b>X</b> value as the smile's threshold varies between lightning conditions and skin color . I used 50 . The Following line is to be modified if smile is not being recognized .

#### "smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, <b>X</b>)"

# Results :
- Not Smiling 
![down](https://image.ibb.co/i8HjAn/not_smiling.png)
- Smiling
![down](https://image.ibb.co/gaV4An/smiling.png)



I am now working on a model which uses CNN .
