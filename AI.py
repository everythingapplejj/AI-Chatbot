# Sample Tester By JJ
# Using openai API. Testing 4 models: davinci, curie, ada, and babbage
import openai
import gtts
from playsound import playsound # now can I add speaking ability !!!!!!
import speech_recognition as sr
openai.api_key = "pk-McpgvqLGlIIuWxjqcbddjVnJiQArhovxRzdcmHwumehxcrYa"
# pk-McpgvqLGlIIuWxjqcbddjVnJiQArhovxRzdcmHwumehxcrYa
openai.api_base = 'https://api.pawan.krd/v1'
# Using the davinci model. Curie vs Ada vs babbage? Which one is better?
# final solution is to use text-davinci-003
def sample_question(input):
  response = openai.Completion.create(
  model="text-davinci-003",
  prompt= input,
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["Human: ", "AI: "]
  )
  return response.choices[0].text.strip()
  
while True:
    r = sr.Recognizer()
    with sr.Microphone() as source:
       print("Say something: ")
       audio = r.listen(source)
    out = r.recognize_google(audio)
    user = out
    print("User:", user)
    sample_word = sample_question(user)
    print("Computer:",sample_word)
    tts = gtts.gTTS(sample_word, lang = "en")
    tts.save("./mp3/say.mp3")
    playsound("./mp3/say.mp3")