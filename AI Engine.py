# Sample Tester By JJ
# Using openai API. Testing 4 models: davinci, curie, ada, and babbage
import openai
import pyttsx3 # now can I add speaking ability !!!!!!
openai.api_key = "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"
engine = pyttsx3.init()
# Using the davinci model. Curie vs Ada vs babbage? Which one is better?
# final solution is to use text-davinci-003
def sample_question(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt = prompt,
        temperature = 0.5,
        max_tokens = 1024,
        top_p=1,
        frequency_penalty = 0,
        presence_penalty = 0
    )
    return response.choices[0].text.strip()
while True:
    user = input("User:")
    engine.say(sample_question(user))
    print("Computer:",sample_question(user))
    engine.runAndWait()