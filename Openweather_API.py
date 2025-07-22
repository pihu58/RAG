import requests
url = "https://api.openweathermap.org/data/2.5/weather?q=Delhi&appid=apikey&units=metric"

def get_weather():
    try:
        response = requests.get(url, verify=False)
        data = response.json()
        return data
    except:
        return "API call failed. Please check your API key or network connection."

#print(get_weather())
