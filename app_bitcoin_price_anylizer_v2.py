
# based on https://www.youtube.com/watch?v=-C4FCxP-QqE

#'''
#you need to install
#!pip install openai
#!pip install requests
#!pip install streamlit

#'''


import openai
import json
import requests
import streamlit as st
import os

# When developing your app locally, add a file called secrets.toml 
# in a folder called .streamlit at the root of your app repo, 
# and copy/paste your secrets into that file. 

# Access your secrets as environment variables or by querying the st.secrets dict. 
# For example, if you enter the secrets from the section above, 
# the code below shows you how you can access them within your Streamlit app.

# Everything is accessible via the st.secrets dict:
# st.write("DB username:", st.secrets["db_username"]) # DB username from secrets is printed
# st.write("DB password:", st.secrets["db_password"]) # DB password from secrets is printed
# st.write("My cool secrets:", st.secrets["my_cool_secrets"]["things_i_like"]) # Other secrets 

# Root-level secrets are also accessible as environment variables.
# import os

# Checks if the environment variable and the secret from st.secrets are the same
#st.write(
#   "Has environment variables been set:",
#   os.environ["db_username"] == st.secrets["db_username"],
#)

# !!!!And the root-level secrets are also accessible as environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set this to `True` if you need GPT4. If not, the code will use GPT-3.5.
GPT4 = True

# '''
# Yes, your code looks correct. It defines a function called `BasicGeneration` that takes a user prompt as input.
# Inside the function, you create a system message to set the context of
# an expert crypto trader with more than 10 years of experience.
#
# Then, you make a request to the OpenAI API using the `openai.ChatCompletion.create()` method.
# You pass in the model version, the messages (including the system message and the user prompt),
# and retrieve the completion choice.
#
# Finally, you return the content of the completion choice using `completion.choices[0].message.content`.
#
# Overall, this code sets up the conversation with the system message and user prompt, and retrieves the generated response.
# '''

# '''
# In the code you provided, the role can be either "system" or "user" for the messages in the conversation.
# The role "system" is typically used for providing instructions or contextual information to the model,
# while the role "user" represents the user's input or prompt.
# '''


class ChatGPTConversator:
    # This class helps me keep the context of a conversation. It's not
    # sophisticated at all and it simply regulates the number of messages in the
    # context window.

    # You could try something much more involved, like counting the number of
    # tokens and limiting. Even better: you could use the API to summarize the
    # context and reduce its length.

    # But this is simple enough and works well for what I need.

    # messages = None — This line declares a global variable named messages and assigns it a value of None.

    # The __init__ Method: This is a constructor in Python classes. 
    # It gets automatically called when a new instance of the class is created. 
    # self is a reference to the newly created object.
   
    # i took this recommended class from https://colab.research.google.com/drive/13c_pwUNkcAb04bimpQG95xb3KQ3txKSd?usp=sharing

    messages = []

 
    def __init__(self):
        # Here is where you can add some personality to your assistant, or
        # play with different prompting techniques to improve your results.
        # i keep it empty for now
        pass


    def answer(self, userPrompt, systemContent=None):
        try:
            # first we insert user prompt
            self._update("user", userPrompt)

            if systemContent is not None:
                ChatGPTConversator.messages.insert(0, {"role": "system", "content": systemContent})

            response = openai.ChatCompletion.create(
                model="gpt-4" if GPT4 else "gpt-3.5-turbo",
                messages=ChatGPTConversator.messages,
                temperature=0.2,
                max_tokens=2000
            )

            # this one probably to keep message history
            self._update("assistant", response.choices[0].message.content)
   
            return response.choices[0].message.content

        except Exception as e:
            print("Error occurred: ", str(e))
            return None

    def _update(self, role, content):
        ChatGPTConversator.messages.append({
            "role": role,
            "content": content,
        })

        # This is a rough way to keep the context size manageable.
        '''
        ChatGPTConversator.messages.pop(0) — This line takes the message at index 0
        (the oldest message, given a typical chat scenario where new messages are appended to the end of the list) 
        and removes it from the messages list.
        '''
        if len(ChatGPTConversator.messages) > 20:
            ChatGPTConversator.messages.pop(0)
            
# test are

# Here is where I put my prompt.
my_content = """
You are a helpful, polite, old English assistant. 
Answer the user prompt with a bit of humor.
"""

my_prompt = """
What should I wear to a fancy restaurant?
"""

# This could be uncommented for testing in terminal output
#Create a new instance of `Conversation` whenever you want to clear the context."""
# conversation = ChatGPTConversator()

# answer = conversation.answer(my_prompt, my_content)
# print(answer)

# I can now ask a question and the API will know what happened before.

# answer = conversation.answer("What about an overshirt?")
# print(answer)

# this function i uses before but now i use class above            
def BasicGeneration(userPrompt, systemContent=None):
    try:

        messages=[
                {"role": "user", "content": userPrompt}
            ]

       # example. "content": "You are an expert crypto trader with more than 10 years of experience."

        if systemContent is not None:
            messages.insert(0, {"role": "system", "content": systemContent})


        completion = openai.ChatCompletion.create(
            model="gpt-4" if GPT4 else "gpt-3.5-turbo",
            messages=messages,
            # Adjust the temperature here.
            # Higher values like 0.8 make the output more random, while lower values like 0.2 make it more focused and deterministic.
            temperature=0.2,
            # Adjust the max tokens here (default is 100, controls the length of the generated response)
            max_tokens=2000
        )

        return completion.choices[0].message.content

    except Exception as e:
        # Handle the exception in an appropriate way
        print("Error occurred: ", str(e))
        return None

# this code is for testing this function. Will be not shown on web page
test_prompt = 'please explain what is Bitcoin to 7 year old kid'
test_response = BasicGeneration(test_prompt)
print(test_response)


def GetBitCoinPrices():
    try:
        # Define the API endpoint and query parameters for Bitcoin for 30 days
        url = "https://coinranking1.p.rapidapi.com/coin/Qwsogvtv82FCd/history"
        querystring = {
            "referenceCurrencyUuid": "yhjMzLPhuIDl",
            "timePeriod": "30d"
        }
        # Define the request headers with API key and host
        headers = {
            "X-RapidAPI-Key": "cfd249bff0mshc564006d90b2713p1591fajsn3cf236f2ef42",
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }
        # Send a GET request to the API endpoint with query parameters and headers
        response = requests.request("GET", url, headers=headers, params=querystring)

        # Check the response status code - if it's not 200, raise an error
        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}")

        # Parse the response data as a JSON object
        JSONResult = json.loads(response.text)
        
        # Extract the "history" field from the JSON response
        history = JSONResult["data"]["history"]
        
        # Sorting the 'history' from the least to the most recent timestamp
        history.sort(key=lambda x: x['timestamp'])

        # Extract the "price" field from each element in the "history" array
        # Join the list of prices into a comma-separated string
        
        prices_string = ','.join([change["price"] for change in history])

        # Return the comma-separated string of prices
        return prices_string
    except Exception as e:
        # Print the error message if an error occurred
        print(f"An error occurred: {str(e)}")
        return None

def app():
    st.title('Bitcoin Analyzer With ChatGPT')
    st.subheader(
        'We are retrieving daily Bitcoin price for the last 30 days')

    if st.button('Analyze'):
        with st.spinner('...'):
            try:
                bitcoinPrices = GetBitCoinPrices()
                st.success('Done Getting Bitcoin Prices!')
            except Exception as e:
                st.error(f'Failed to get Bitcoin prices: {str(e)}')
                return
        with st.spinner('Analyzing Bitcoin Prices...'):

            chatGPTSystemContent = "You are an expert  crypto trader with mode then 10 years of experience"
            chatGPTUserPrompt = f"""
                        I will provide you with a list of bitcoin prices for the last 30 days from earliest to the latest.
                        Can you please provide me with a technical analysis
                        of Bitcoin based on these prices. Here is what I want:
                        Price Overview,
                        What is the today price of Bitcoin
                        Moving Averages ,
                        Relative Strength Index (RSI),
                        Moving Average Convergence Divergence (MACD),
                        Advice and Suggestion,
                        Do I buy or sell?
                        Please be as detailed as much as you can, and explain in a way any beginner can understand.
                        and make sure to use headings
                        Here is the price list i mentioned above: {bitcoinPrices}"""

            try:
                conversator = ChatGPTConversator()
             
                analysis = conversator.answer(chatGPTUserPrompt, chatGPTSystemContent)
                st.text_area("Analysis", analysis, height=500)
                st.success('Done Analyzing Bitcoin Prices!')
            except Exception as e:
                st.error(f'Failed to generate analysis: {str(e)}')
                return

app()

# '''
# Here's the breakdown of the script you provided. This seems to be a Streamlit application for analyzing Bitcoin prices using a ChatGPT model.
#
# The function `app()` is defined. This function would be running when you start your streamlit app.
#
# Inside the `app` function:
#
#    - It sets the title of the Streamlit app to 'Bitcoin Analyzer With ChatGPT' using `st.title()`.
#
#    - It sets a subheader saying 'Implemented by Alex & Brian!' using `st.subheader()`.
#
#    - A button labeled 'Analyze' is created using `st.button()`. If this button is clicked, it will trigger the block of code inside the if statement.
#
# When the button is clicked:
#
#    - A spinner is displayed with the message 'Getting Bitcoin Prices...' using `st.spinner()`. While this spinner is displayed, it tries to get Bitcoin prices.
#
#      - The function `GetBitCoinPrices()` is called to get the prices. If success, it displays a message 'Done!' using `st.success()`, and the resulting prices are stored
#        in the `bitcoinPrices` variable.
#
#      - If an exception occurs while getting prices (maybe due to network errors or API failures), it catches that with `except Exception as e:`, logs an error message 
#        with `st.error()`, and terminates the current execution using `return`.
#
#    - After getting Bitcoin prices successfully, another spinner is displayed with the message 'Analyzing Bitcoin Prices...'.
#
#      - A system content and user prompt are prepared for the ChatGPT model.
#
#      - Then it calls the `BasicGeneration(chatGPTUserPrompt, chatGPTSystemContent)` function which presumably uses a GPT-3 model to generate analysis based on the 
#        bitcoin prices obtained earlier, and the generated analysis is displayed in a text area.
#
#      - If an exception occurs during the generation (maybe due to a problem with the ML model or other issues), it catches that with `except Exception as e:`, 
#        logs an error message, and terminates the program flow.
#
# In summary, it's a simple Streamlit application that fetches Bitcoin price data, analyzes it via a ChatGPT ML model when user clicks the 'Analyze' button, 
# and then displays the result or any errors encountered in the process.
#

