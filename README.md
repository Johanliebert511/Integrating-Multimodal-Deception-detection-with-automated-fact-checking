# Integrating-Multimodal-Deception-detection-with-automated-fact-checking (Pytorch)

## Flowchart of deception detection
The overall flowchart for deception detection is illustrated below. We combine many features into a vector and then apply SVM to classification.
<p align="center">
 <img src="https://github.com/come880412/Deception_detection/blob/main/img/Flowchart%20.png" width=50% height=50%>
</p>

## Face alignment
Because the face can be many angles, we need to align the face before using it.
<p align="center">
 <img src="https://github.com/come880412/Deception_detection/blob/main/img/face%20alignment.png" width=50% height=50%>
</p>

## User instrctions
Our deception detection system comprises four parts：
1. 3D landmarks displacement
2. Emotion Unit
3. Action Unit
4. Emotion Audio unit
5. Fact checking unit

### Install Packages
Please see the ```requirements.txt``` for more details.

## Pre-trained models
- Please download the pre-trained models before you run the code
<https://drive.google.com/drive/folders/1izJQF5bcbqNdDpYAdFvLCG0qJnHlicAg?usp=sharing>

## Dataset
### Real-life trial dataset:
121 videos including 61 deceptive videos and 60 truth videos [related paper](https://web.eecs.umich.edu/~zmohamed/PDFs/Trial.ICMI.pdf)
### Bag-of-lies dataset:
325 videos including 162 deceptive videos and 163 truth videos [related paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Gupta_Bag-Of-Lies_A_Multimodal_Dataset_for_Deception_Detection_CVPRW_2019_paper.pdf)
### MSPL-YTD dataset:
145 videos including 62 deceptive videos and 83 truth videos

- Note: If you would like access to the above datasets, please contact the authors who provided the dataset, respectively.
## GUI demo
![image](https://github.com/Johanliebert511/Integrating-Multimodal-Deception-detection-with-automated-fact-checking/blob/main/img/Demo.png)
![image](https://github.com/Johanliebert511/Integrating-Multimodal-Deception-detection-with-automated-fact-checking/blob/main/img/audio.png)
## Inference
```python=
python lie_GUI.py
```
