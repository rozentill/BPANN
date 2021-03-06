## Handwriting Number Recognition

Written by : Python  
Prerequisite : pip, Numpy, PIL(pillow), matplotlib  
Dataset : [MINIST](http://http://yann.lecun.com/exdb/mnist/)

#### Introduction
---
This project is an example of my BPANN package, also it is an assignment of the course **ES901**(智能信息处理). I used BP network to train the **MINIST** dataset and then tested it with the test data of **MINIST** and the images generated by program [*drawNumber.py*](#).

#### How to Play

---
I wrote some interfaces thus you can play this project and try to recognize the images on your own.  

**Environment**  
Before you start, you must make sure your environment has installed : Python 2.7, Numpy, PIL(now you can use pillow instead), Matplotlib. Then you should download this project and go into this **number** directory.   

**Generate Images**  
You can generate images by executing the following command :
```shell
$python drawNumber.py
```
Then it will show you some words :
```shell
Please enter the file name you want to save for this randomly generated image :
```
You just need to input a file name such as 001, 002 and some other names you like. And you can see the following message :  
```shell
The image is saved successfully.
```
After that, you will find the image already generated like this :  ![image](https://raw.githubusercontent.com/rozentill/BPANN/master/example/number/numbers/002.jpg)
stored in directory **numbers**. Now you can test them under the following guide.

**Recognize the Images**  
Similar to the above, you just need to type a command at first :
```shell
$python numberRec.py
```
Actually it will read the model stored in **model.txt** and set the neural network.
Now there is a similar sentence :
```shell
Please enter the file path you want to recognize :
```
You need to type a file name(e.g. 001) exists in **numbers** directory.  
Then you can get the recognization result like the following :
```shell
The image is recognized as :0
```

#### Training Process and Result
---
1. **Process the data**  
Actually the data of MINIST can not be used to train directly since it is not a binarized image data. So I binarized and normalized them by detecting whether the value of a pixel is bigger than 136. If it is, set it as 1, otherwise set it as 0. The results are as follows :  
![image](https://raw.githubusercontent.com/rozentill/BPANN/master/example/number/handwriting/train_gray.png)
![image](https://raw.githubusercontent.com/rozentill/BPANN/master/example/number/handwriting/train_binary.png)  
2. **Train by BP Network**  
I test several times and find the **15** hidden layer nodes and **0.05** learning rate may be good for the training. And I only train one iteration since the result is already good. The command is as follow :
```shell
$python numberTrain.py
```
After this, the model will be saved automatically in **model.txt**.  
3. **Training Results**  

Learning rate\No. of Hidden Nodes | 11 | 13 | 15 | 17 | 19
---|---|---|---|---|---
0.05 | 74.98% | 85.12% | 85.23% | 84.71% | 63.59%

#### Files Explanation
---
- net.py : implement a neural network class.
- function.py : implement two activation functions and can be chosen by users.
- minist.py : implement a class for MINIST dataset which can be used to access the dataset file simply.
- numberTrain.py : the training program, it also stores the result in the model.txt.
- numberRec.py : load the weights stored in model.txt and recognize an image.
- drawNumber.py : used to generate a digit image automatically.

#### In the Future
---
I wish this project can recognize more characters like letters and can be more stronger in the dirty data. Also I wish it can be a package which allows people to attack the CAPTCHA.
