# Inferring soundscape clips for LLM 

This part aims to convert the soundscape audio clips into the predicted audio event probabilities, the acoustic scene labels, and the ISOP, ISOE, and PAQ 8D AQ values.

This part bridges the acoustic model and the language model, organising the output of the acoustic model in preparation for the input of the language model.

## 1. Data preparation

- Place the log Mel features file from the `Feature_log_mel` directory into the `Dataset_mel` directory

- Place the ISO 532-1 loudness features file from the `Feature_loudness_ISO532_1` directory into the `Dataset_wav_loudness` directory
 

## 2. Run the inference script

- cd application

- python Inference_for_LLM.py

- The results, which will be fed into the LLM, will be automatically saved into the corresponding directories: `SoundAQnet_event_probability`, `SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs`

- There are two slightly different SoundAQnet models in the `system/model` directory; please feel free to use them
	- SoundAQnet_ASC96_AEC94_PAQ1027.pth
	- SoundAQnet_ASC96_AEC94_PAQ1039.pth
	- SoundAQnet_ASC96_AEC94_PAQ1041.pth
	- SoundAQnet_ASC96_AEC95_PAQ1052.pth
 

## 3. Inference with other models

This part of the model uses SoundAQnet to infer the values of audio events, acoustic scenes, and emotion-related PAQ AQs. 

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
Audio event probability matrix: [5.6956115e-04 9.9856347e-01 3.6731756e-03 4.0320985e-02 3.3493016e-02
 2.8794003e-01 4.9668720e-01 4.8883647e-01 9.9535048e-01 6.8625987e-01
 7.9891986e-01 9.9824548e-01 1.6407473e-01 1.6404054e-01 9.9983203e-01]
Acoustic scene: park
ISOP and ISOE: 0.42336637	-0.10498028
PAQ 8D AQs: 4.108083	3.0842495	2.0122054	3.4061074	2.813982	4.00237	1.8789594	2.4632325
```
  

