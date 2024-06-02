import openai
import pickle, os, sys
from datetime import datetime

# waves
openai.api_key = "YOUR_API_KEY_HERE"

output_file = os.path.join(os.getcwd(), 'dict_data_labels.pickle')
with open(output_file, 'rb') as f:
    data_dict = pickle.load(f)

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

 

filename = os.path.basename(__file__).split('.py')[0]
print_log_file = os.path.join(os.getcwd(), filename + '_print.log')

# prompt1
def get_completion_from_messages(messages,
                                 model="gpt-3.5-turbo",
                                 # model="gpt-4",
                                 temperature=1,
                                 max_tokens=250):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]


for key, value in data_dict.items():
    print(key, value)

    event_labels_original = value['event_labels']
    emotion = value['emotion']
    # print(emotion)
    # ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']
    event_probability_original = value['event']
    # Find indices where the label is not "Animal"

    print("labels:", event_labels_original)

    thresholds = [0.4, 0.3, 0.2]
    # threshold=0
    if any(i > 0.4 for i in event_probability_original):
        threshold = thresholds[0]
    elif any(i > 0.3 for i in event_probability_original):
        threshold = thresholds[1]
    elif any(i > 0.2 for i in event_probability_original):
        threshold = thresholds[2]

    # if any of the event_probability is too low, set it to 0
    for i in range(len(event_probability_original)):
        if event_probability_original[i] < threshold:
            event_probability_original[i] = 0
    print("p:", event_probability_original)
    scene = value['scene']

    sorted_filtered_events = sorted(
        [(prob, label) for prob, label in zip(event_probability_original, event_labels_original) if prob != 0],
        reverse=True)

    # Unzip the sorted and filtered list into two lists
    event_probability, event_labels = zip(*sorted_filtered_events)
    # Convert the tuples back to lists (optional, depending on your requirements)
    event_probability = list(event_probability)
    event_labels = list(event_labels)
    # if the label is Animal, delete the label and probability
    # Keep to two decimal places
    event_probability = [round(i, 2) for i in event_probability]

    print(event_labels)
    print(event_probability)
    PAQ_8_values = value['PAQ_8_values']
    print(PAQ_8_values)

    attitude = []
    for ratings in PAQ_8_values:
        if ratings <= 1.5:
            ratings = "Strongly Disagree"
        elif ratings <= 2.5 and ratings > 1.5:
            ratings = "Disagree"
        elif ratings <= 3.5 and ratings > 2.5:
            ratings = "Neutral"
        elif ratings <= 4.5 and ratings > 3.5:
            ratings = "Agree"
        elif 4.5 < ratings:
            ratings = "Strongly Agree"
        attitude.append(ratings)

    filtered_events = [(ATT, EMO) for ATT, EMO in zip(attitude, emotion) if ATT != 'Neutral']
    if len(filtered_events) == 0:
        filtered_events = [(attitude[0], emotion[0])]
    print(filtered_events)
    # Unzip the sorted and filtered list into two lists
    attitude, emotion = zip(*filtered_events)

    # Convert the tuples back to lists (optional, depending on your requirements)
    attitude = list(attitude)
    emotion = list(emotion)

    # if the label is Animal, delete the label and probability
    print(attitude)
    print(emotion)
    Affective_Responses_list = []
    for ATT, EMO in zip(attitude, emotion):
        Affective_Responses = f"I {ATT} with {EMO}"
        Affective_Responses_list.append(Affective_Responses)
    print(Affective_Responses_list)

    system_message_1 = f"""

    A  soundscape is defined as the an acoustic environment as perceived or experienced and/or understood by a person or people in context, accompanying physiological and psychological responses.

    Consider this as a guided task to create an soundscape description:

    As an expert in soundscape analysis, your task is to write a description based on provided affective responses and auditory events.

    To proceed, you are in this {scene}, where you hear sound events, including {event_labels}, each occurring with a probability of {event_probability}, ordered with the decreasing probability of being heard.

    step 1. According to the {event_labels} and its corresponding probability of {event_probability} happening in this {scene},
      Identify which sound events will be present and describe the auditory scenario according to their occurrance,
      If the {event_probability} is too low, you can ignore it, please do not mention the probability numbers {event_probability}  directly in the description.
       Please do not add or relate any new sound events which is not belong to the {event_labels} to the description. 
       Please do not describe the sound events details, 
       for example, vehicle sounds does not mean "The constant hum of engines and occasional honking of horns", please just write the sound events itself.


    step 2. Describe your feelings based on the ratings {Affective_Responses_list} on this soundscape:

    Now, your task is to write a soundscape description within 200 tokens which summerize the steps above, please describe it in lay language, keeping it simple and concise.

    example:
    In this {scene},.....
    """

    messages = [
        {'role': 'system',
         'content': system_message_1}
    ]

    final_response = get_completion_from_messages(messages)

    print(final_response)
    print("###############")
