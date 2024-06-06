import openai
import pickle, os, sys


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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


def main(argv):
    # waves
    openai.api_key = "YOUR_API_KEY_HERE"

    output_file = os.path.join(os.getcwd(), 'Dict_data_labels.pickle')
    with open(output_file, 'rb') as f:
        data_dict = pickle.load(f)

    for key, value in data_dict.items():
        print('Soundscape audio clip:', key.split('_scene_PAQ.txt')[0])

        event_labels_original = value['event_labels']
        emotion = value['PAQ8ARs']
        event_probability_original = value['event']
        thresholds = [0.4, 0.3, 0.2]
        if any(i > 0.4 for i in event_probability_original):
            threshold = thresholds[0]
        elif any(i > 0.3 for i in event_probability_original):
            threshold = thresholds[1]
        elif any(i > 0.2 for i in event_probability_original):
            threshold = thresholds[2]
        for i in range(len(event_probability_original)):
            if event_probability_original[i] < threshold:
                event_probability_original[i] = 0
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

        PAQ_8_values = value['PAQ_8_values']
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
        attitude, emotion = zip(*filtered_events)

        # Convert the tuples back to lists (optional, depending on your requirements)
        attitude = list(attitude)
        emotion = list(emotion)

        Affective_Responses_list = []
        for ATT, EMO in zip(attitude, emotion):
            Affective_Responses = f"I {ATT} with {EMO}"
            Affective_Responses_list.append(Affective_Responses)

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

        print('Soundscape caption:', final_response)


# Demo:
# In this park, you will hear Bird and Animal sounds, with occasional Human sounds and Speech in the background. The atmosphere feels pleasant and calm, perfect for relaxation and enjoying nature.


# In this park, you may hear bird songs, occasional animal sounds, human activities, and snippets of speech. The overall atmosphere feels pleasant and calm, with a touch of natural tranquility despite some human presence.

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)



