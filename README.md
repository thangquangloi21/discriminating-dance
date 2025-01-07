## discriminating-dance
Jumping motion recognition via video is a significant contribution because it considerably impacts intelligent applications and will be widely adopted in life. This method can be used to train future dancers using innovative technology. Challenging poses will be repeated and improved over time, reducing the strain on the instructor when performing multiple times. Dancers can also be recreated by removing features from their images. To recognize the dancers’ moves, check and correct their poses, and another important aspect is that our model can extract cognitive features for efficient evaluation and classification, and deep learning is currently one of the best ways to do this for short-form video features capabilities. In addition, evaluating the quality of the performance video, the accuracy of each dance step is a complex problem when the eyes of the judges cannot focus 100% on the dance on the stage. Moreover, dance on videos is of great interest to scientists today, as technology is increasingly developing and becoming useful to replace human beings. Based on actual conditions and needs in Vietnam. In this paper, we propose a method to replace manual evaluation, and our approach is used to evaluate dance through short videos. In addition, we conduct dance analysis through short-form videos, thereby applying techniques such as deep learning to assess and collect data from which to draw accurate conclusions. Experiments show that our assessment is relatively accurate when the accuracy and F1-score values are calculated. More than 92.38% accuracy and 91.18% F1-score, respectively. This demonstrates that our method performs …

This source code has been published in the [magazine](http://www.proceedings.spiiras.nw.ru/index.php/sp/article/view/16027). If you use this source code, please cite the above document


## Install
install python

This source code is compatible with [python 3.7](https://www.python.org/downloads/) and above, so please pay attention to installing the correct compatible version

install setup.txt
 ```
pip install -r setup.txt --user
 ```

## Steps to train the model:
I. Data collection:
- Collect hot trend dances on tiktok. Each trend will have 8 to 10 videos per trend

II. Data preprocessing:
- After collecting data, use Make_data.py to browse through each folder to bring the movement data to a .csv file according to each trend.

III. Model training:
- Based on the available model, we see that there are 3 main models:
• RNN (Recurrent neural network)
• LSTM (Long Short-Term Memory)
• GRU (Gated Recurrent Unit)
- replace the inp_dance variables to point to the folder containing the processed video data
- Then run each model to train
Model RNN
Model LSTM
Model GRU
- And must change the label name according to the labels configured in the train model section

This is the code in the models

This is the code in inference.py
IV. Evaluate model	
Model/Value	RNN	LSTM	GRU
accuracy	0.9623	0.9738	0.9694
loss	0.1409	0.0718	0.0852
F1 score	0.9542	0.9715	0.9621

V. Deployment testing
- Use the inference.py file to use the desired model, change the model and cap variables according to the model you want to deploy and the video path you want to use.
- 1 example of using RNN model to detect trend
