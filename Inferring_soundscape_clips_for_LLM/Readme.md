# Inferring soundscape clips for LLM 

This part aims to convert the soundscape audio clips into the predicted audio event probabilities, the acoustic scene labels, and the ISOP, ISOE, and PAQ 8D AR values.

This part bridges the acoustic model and the language model, organising the output of the acoustic model in preparation for the input of the language model.

## 1. Data preparation

- Place the log Mel features file from the `Feature_log_mel` directory into the `Dataset_mel` directory

- Place the ISO 532-1 loudness features file from the `Feature_loudness_ISO532_1` directory into the `Dataset_wav_loudness` directory
 

## 2. Run the inference script

- cd application

- python Inference_for_LLM.py

- The results, which will be fed into the LLM, will be automatically saved into the corresponding directories: `SoundAQnet_event_probability`, `SoundAQnet_scene_ISOPl_ISOEv_PAQ8DARs`

- There are two slightly different SoundAQnet models in the `system/model` directory; please feel free to use them
	- SoundAQnet_PAQ1054.pth
	- SoundAQnet_PAQ1075.pth
 

## 3. Inference with other models

This part of the model uses SoundAQnet to infer the values of audio events, acoustic scenes, and emotion-related PAQ ARs. 

If you want to replace SoundAQnet with another model and generate the soundscape captions corresponding to that model, 
- replace `using_model = SoundAQnet` in `Inference_for_LLM.py` with the code for that model, 
- and place the corresponding trained model into the `system/model` directory.
  

## 4. Demonstration

```python
python Inference_for_LLM.py
-----------------------------------------------------------------------------------------------------------
Loading data time: 0.111 s
Inference audio clip mel:  (1, 3001, 64)
Inference audio clip loudness:  (1, 15000, 1)
Number of 1 audio clip(s) in inference

Soundscape audio clip: fold_1_participant_00056_stimulus_13
Audio event probability matrix: [0.00153319 0.99596703 0.00343861 0.0276242  0.03176526 0.20033626
 0.1959963  0.18441455 0.9944666  0.71280897 0.80586416 0.9958527
 0.17599502 0.44230244 0.9998097 ]
Acoustic scene: park
ISOP and ISOE: 0.39127547	-0.11820114
PAQ 8D ARs: 3.9432704	2.833214	1.8872974	3.217749	2.985929	3.9750671	2.0713344	2.7494504
```
  

