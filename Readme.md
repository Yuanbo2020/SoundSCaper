# LLM-SSD: Large Language Model and sound emotional quality network-based SoundScape Describer for soundscape caption

## Introduction

### LLM-SSD Training steps

1) Download and prepare the dataset in [Dataset_ARAUS](Dataset_ARAUS); ARAUS can be downloaded [ARAUS_repository](https://github.com/ntudsp/araus-dataset-baseline-models/tree/main). The labels of our annotated acoustic scenes and audio events are in [Dataset_ARAUS](Dataset_ARAUS).


2) Use the code in [Feature_log_mel](Feature_log_mel) to extract log mel acoustic features; use the code in [Feature_loudness_ISO532_1](Feature_loudness_ISO532_1) to extract the ISO 532-1:2017 standard loudness features.

3) Use the code in [Model_SoundEQnet](Model_SoundEQnet) to train SoundEQnet.
  
4) In [LLM_scripts](LLM_scripts), read the audio scene, audio events, and PAQ 8-dimensional affective response corresponding to the test audio predicted by the trained SoundEQnet, and then output the corresponding soundscape descriptions. <br> Please fill in your OpenAI username and password in [LLM_scripts](LLM_scripts).

### Expert evaluation of LLM-SSD soundscape caption quality

[Human_assessment](Human_assessment) contains 

1) a call for experiment; 

2) assessment raw materials: assessment dataset; participant instruction file; local and online questionnaires; 

3) assessment statistical results from a jury composed of 16 audio/soundscape experts.

### LLM-SSD One Run

[One_Run](One_Run) provides the scripts to convert target audio clips directly into soundscape descriptions for easy inference-only use.

If you want to skip the tedious training steps and use LLM-SSD directly, go directly to [One_Run](One_Run).

Please fill in your OpenAI username and password in LLM_scripts.

### Other models
  
The trained models of the other 7 models in the paper have been attached to their respective folders. 

If you want to train them yourself, please follow the LLM-SSD training steps.

<br>

## Figure

### 1) Overall framework of the soundscape describer LLM-SSD

<h3 align="center"> <p></p></h3>
<div align="center">
<img src="Figure/LLM_SSD.png" width=100%/> 
</div>  

<br>
### 2) The acoustic model SoundEQnet simultaneously models acoustic scene (AS), audio event (AE), and emotion-related affective response (AR)

<h3 align="center"> <p></p></h3>
<div align="center">
<img src="Figure/SoundEQnet.png" width=100%/> 
</div> 

<br>
### 3) Process of the LLM part in the soundscape describer LLM-SSD
<h3 align="center"> <p></p></h3>
<div align="center">
<img src="Figure/LLM_part.png" width=100%/> 
</div> 



## Run models

```1) Unzip the Dataset under the application folder```

```2) Unzip the Pretrained_models under the application folder```

```3) Enter the application folder: cd application```

