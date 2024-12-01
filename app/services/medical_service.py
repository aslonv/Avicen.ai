import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import json
from bs4 import BeautifulSoup

# Load a model for generating embeddings
model1= SentenceTransformer('all-MiniLM-L6-v2')

keys = ["Drugs, Sumplements, and Medicines", "Medical Encyclopedia: Diseases, Symptoms, treatments, procedures", "Medical tests and results"]
idx = ["drugs", "encyclopedia", "test"]
embeddings1 = model1.encode(keys)

dimension1 = embeddings1.shape[1]  # Assuming embeddings are 2D array
index1 = faiss.IndexFlatL2(dimension1)
index1.add(embeddings1)

def retrieve1(query, k=1):
    query_embedding = model1.encode([query])
    distances, indices = index1.search(query_embedding, k)
    return indices[0]

health = json.load(open("health.json"))
health["drugs"]["base_link"] = "https://medlineplus.gov/druginfo/"
health["encyclopedia"]["base_link"] = "https://medlineplus.gov/ency/"
health["test"]["base_link"] = ""
for key in health.keys():
    health[key]["model"] = SentenceTransformer('all-MiniLM-L6-v2')
    health[key]["embeddings"] = health[key]["model"].encode(health[key]["names"])
    health[key]["dimension"] = health[key]["embeddings"].shape[1]
    health[key]["index"] = faiss.IndexFlatL2(health[key]["dimension"])
    health[key]["index"].add(health[key]["embeddings"])

def retrieve2(query,key, k=1):
    query_embedding = health[key]["model"].encode([query])
    distances, indices = health[key]["index"].search(query_embedding, k)
    return indices[0]


def retrieve(query):
    index = retrieve1(query)[0]
    return (health[idx[index]]["names"][retrieve2(query, idx[index])[0]],health[idx[index]]["links"][retrieve2(query, idx[index])[0]])




def parse_output(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    article = soup.find('article')
    if article:
        text_content = '. '.join(article.stripped_strings)
        return text_content
    else:
        return html_content

users = {}
users["0"] = 'Patient Profile:\n    - Name: Emily Johnson\n    - Age: 63\n    - Gender: Female\n    - the patient has the following Known Conditions: Hypertension\n    - the pattient is Currently taking the Medication: Sertraline\n    - The patient has the following allergies: Peanuts\n    - Smoking Status: Never smoker\n    - Family History: No significant history\n    - Symptoms: Blurred vision'

import requests
from groq import Groq
import os
os.environ['GROQ_API_KEY'] = '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'
question = "I think i have a headache should i take an aspirin?"#"I can't distinguish colors, what can I do?"#"What are the symptoms of appendicitis?"
def query_lm(question, user,max_tokens=8000):
    sources = []
    contexts = retrieve(question)
    context = ""
    bl = health[idx[retrieve1(question)[0]]]["base_link"]
    if type(contexts) == list:
        for c in contexts:
            content = parse_output(requests.get(bl+c[1]).text)  # Assuming 'c[1]' is the URL
            context += content + "\n\n"
            sources.append(c[1])
    else:
        context = parse_output(requests.get(bl + contexts[1]).text)
        sources.append(contexts[1])
    client = Groq()
    messages=[{"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"}, {"role": "user", "content":
        users[user] },{"role": "system",
                    "content": "You are Dr. Emily Carter, a seasoned and empathetic physician with over 20 years of experience in internal medicine. You've treated thousands of patients, and your approach is always professional, thorough, and considerate. Here are your guidelines:\n\n- **Medical Accuracy**: Base your diagnosis and advice solely on the medical information provided. Avoid speculation or information not given in the context.\n- **Empathy**: Show empathy towards the patient's condition but maintain professionalism.\n- **Clarity**: Use clear, concise language that a layperson can understand. Avoid unnecessary medical jargon unless explaining it.\n- **Questioning**: If additional information is needed for diagnosis or advice, ask follow-up questions politely.\n- **Treatment Suggestions**: Provide general treatment advice or suggest seeing a specialist when appropriate, but do not prescribe medication directly.\n- **Confidentiality**: Treat all patient information as confidential.\n\nHere's what you know about the conditions that may be affecting the patient:\n\n"+context+ "\n\nAnd Here's the patient's profile, consider that this patient might have allergies that may affect any potential treatment recomendations and also verify that the medication the patient is currently taking if any may affect. Profile:\n\n"+users[user] + "\n\nIf the patient asks about a medication or treatment make sure its safe considering the patient's profile. If any allergies pertaining to the users request are present on the users profile simply answer with 'Due to your allergies it is not safe' otherwise make a recomendation based on the efectiveness of the treatment/drug considering the users profile."},{
                    "role": "user",
                    "content": "Answer the question considering the allergies medications i provided in my Profile (if any), " + question
                }]
    completion = client.chat.completions.create(
           model="llama-3.1-70b-versatile",
            messages = messages,
            temperature=1,
            max_tokens=max_tokens,
            top_p=1,
            stream=True,
            stop=None,
    )
    output1 = ""
    for chunk in completion:
        output1 += chunk.choices[0].delta.content or ""
    return output1





import re
import json

def extract_json(text):
    # Regex pattern to find JSON content between curly braces
    pattern = r'\{.*\}'

    # Find all matches, but we'll assume there's only one JSON in the string for simplicity
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            # Verify if the string is valid JSON
            json.loads(json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("The extracted string does not appear to be valid JSON.")
            return None
    else:
        print("No JSON content found in the string.")
        return None

from pdf2image import convert_from_path
import pytesseract

def read_bloodtest(pdf_path):
    client = Groq()
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)

    messages=[{"role": "system", "content": "You are an expert trained on healthcare and biomedical domain, sepcifically reading test results!"}, {"role": "user", "content":
        text },{"role": "system",
                    "content": "Given the test results you output them in a json containing the test name and the test value. Example {'Hemoglobin': 12.5, 'RBC Count':5.2}"},{
                    "role": "user",
                    "content": "This is my test result: " + text
                }]
    completion = client.chat.completions.create(
           model="llama-3.1-70b-versatile",
            messages = messages,
            temperature=1,
            max_tokens=4000,
            top_p=1,
            stream=True,
            stop=None,
    )
    output1 = ""
    for chunk in completion:
        output1 += chunk.choices[0].delta.content or ""

    blood_result = extract_json(output1)

    modelres = ""
    for x,y in blood_result.items():
        modelres += query_lm(f"Are the following resutls ok, if not what do they mean, please explain very briefly? Test: {x} Result: {y})", "0",80)

    messages=[{"role": "system", "content": "You are an expert trained on healthcare and biomedical domain, sepcifically reading test results and conveying the most important attributes of the results to your patients in a clear and concise manner! Given a series of analysis of several results you look at the broad picture and explain to your patients what that picture is and how he/she is doing given those results."}, {"role": "user", "content": "Individual test result analysis please sumarize them clearly and briefly and give me a list of foods to eat to help improve these results: " + modelres }]
    completion = client.chat.completions.create(
               model="llama-3.1-70b-versatile",
                messages = messages,
                temperature=1,
                max_tokens=4000,
                top_p=1,
                stream=True,
                stop=None,
        )
    output1 = ""
    for chunk in completion:
        output1 += chunk.choices[0].delta.content or ""

    foods = [
        {
            "foodname": "Baked Tofu",
            "link": "https://medlineplus.gov/recipes/baked-tofu/",
            "recipe": "90 minutes"
        },
        {
            "foodname": "Beef and Broccoli",
            "link": "https://medlineplus.gov/recipes/beef-and-broccoli/",
            "recipe": "45 minutes"
        },
        {
            "foodname": "Bell Pepper Nachos",
            "link": "https://medlineplus.gov/recipes/bell-pepper-nachos/",
            "recipe": "20 minutes"
        },
        {
            "foodname": "Black Bean Soup",
            "link": "https://medlineplus.gov/recipes/black-bean-soup/",
            "recipe": "60 minutes"
        },
        {
            "foodname": "Cheesy Beef Pasta",
            "link": "https://medlineplus.gov/recipes/cheesy-beef-pasta/",
            "recipe": "50 minutes"
        },
        {
            "foodname": "Cheesy Potato Soup",
            "link": "https://medlineplus.gov/recipes/cheesy-potato-soup/",
            "recipe": "55 minutes"
        },
        {
            "foodname": "Chicken and Black Bean Salsa Burritos",
            "link": "https://medlineplus.gov/recipes/chicken-and-black-bean-salsa-burritos/",
            "recipe": "50 minutes"
        },
        {
            "foodname": "Chicken Creole",
            "link": "https://medlineplus.gov/recipes/chicken-creole/",
            "recipe": "40 minutes"
        },
        {
            "foodname": "Classic Macaroni and Cheese",
            "link": "https://medlineplus.gov/recipes/classic-macaroni-and-cheese/",
            "recipe": "45 minutes"
        },
        {
            "foodname": "Creamy Potato Leek Soup",
            "link": "https://medlineplus.gov/recipes/creamy-potato-leek-soup/",
            "recipe": "60 minutes"
        },
        {
            "foodname": "Crunchy Chicken Nuggets",
            "link": "https://medlineplus.gov/recipes/crunchy-chicken-nuggets/",
            "recipe": "45 minutes"
        },
        {
            "foodname": "Curried Pumpkin Soup",
            "link": "https://medlineplus.gov/recipes/curried-pumpkin-soup/",
            "recipe": "40 minutes"
        },
        {
            "foodname": "Easy Meatballs",
            "link": "https://medlineplus.gov/recipes/easy-meatballs/",
            "recipe": "70 minutes"
        },
        {
            "foodname": "Fiesta Barley Salad",
            "link": "https://medlineplus.gov/recipes/fiesta-barley-salad/",
            "recipe": "60 minutes"
        },
        {
            "foodname": "Fish Salad",
            "link": "https://medlineplus.gov/recipes/fish-salad/",
            "recipe": "50 minutes"
        },
        {
            "foodname": "Fish Tacos",
            "link": "https://medlineplus.gov/recipes/fish-tacos/",
            "recipe": "40 minutes"
        },
        {
            "foodname": "Garden Vegetable Cakes",
            "link": "https://medlineplus.gov/recipes/garden-vegetable-cakes/",
            "recipe": "35 minutes"
        },
        {
            "foodname": "Green Pea Soup",
            "link": "https://medlineplus.gov/recipes/green-pea-soup/",
            "recipe": "20 minutes"
        },
        {
            "foodname": "Green Salad with Peas",
            "link": "https://medlineplus.gov/recipes/green-salad-with-peas/",
            "recipe": "15 minutes"
        },
        {
            "foodname": "Ham and Vegetable Chowder",
            "link": "https://medlineplus.gov/recipes/ham-and-vegetable-chowder/",
            "recipe": "35 minutes"
        },
        {
            "foodname": "Kale and White Bean Soup",
            "link": "https://medlineplus.gov/recipes/kale-and-white-bean-soup/",
            "recipe": "30 minutes"
        },
        {
            "foodname": "Kale Dip",
            "link": "https://medlineplus.gov/recipes/kale-dip/",
            "recipe": "15 minutes"
        },
        {
            "foodname": "Lentil Confetti Salad",
            "link": "https://medlineplus.gov/recipes/lentil-confetti-salad/",
            "recipe": "35 minutes"
        },
        {
            "foodname": "Lentil Taco Filling",
            "link": "https://medlineplus.gov/recipes/lentil-taco-filling/",
            "recipe": "40 minutes"
        },
        {
            "foodname": "Meatball Soup",
            "link": "https://medlineplus.gov/recipes/meatball-soup/",
            "recipe": "65 minutes"
        },
        {
            "foodname": "Minestrone Soup",
            "link": "https://medlineplus.gov/recipes/minestrone-soup/",
            "recipe": "60 minutes"
        },
        {
            "foodname": "No-Yeast Pizza Crust",
            "link": "https://medlineplus.gov/recipes/no-yeast-pizza-crust/",
            "recipe": "35 minutes"
        },
        {
            "foodname": "One-Pan Chicken Alfredo",
            "link": "https://medlineplus.gov/recipes/one-pan-chicken-alfredo/",
            "recipe": "60 minutes"
        },
        {
            "foodname": "Oven Baked Salmon",
            "link": "https://medlineplus.gov/recipes/oven-baked-salmon/",
            "recipe": "20 minutes"
        },
        {
            "foodname": "Pasta Ratatouille",
            "link": "https://medlineplus.gov/recipes/pasta-ratatouille/",
            "recipe": "50 minutes"
        },
        {
            "foodname": "Quinoa Salad",
            "link": "https://medlineplus.gov/recipes/quinoa-salad/",
            "recipe": "30 minutes"
        },
        {
            "foodname": "Ranch Dip",
            "link": "https://medlineplus.gov/recipes/ranch-dip/",
            "recipe": "5 minutes"
        },
        {
            "foodname": "Ranch Dressing",
            "link": "https://medlineplus.gov/recipes/ranch-dressing/",
            "recipe": "5 minutes"
        },
        {
            "foodname": "Rice Bowl Southwestern Style",
            "link": "https://medlineplus.gov/recipes/rice-bowl-southwestern-style/",
            "recipe": "25 minutes"
        },
        {
            "foodname": "Salmon Patties",
            "link": "https://medlineplus.gov/recipes/salmon-patties/",
            "recipe": "30 minutes"
        },
        {
            "foodname": "Salmon Salad Mix",
            "link": "https://medlineplus.gov/recipes/salmon-salad-mix/",
            "recipe": "10 minutes"
        },
        {
            "foodname": "Salsa Chicken",
            "link": "https://medlineplus.gov/recipes/salsa-chicken/",
            "recipe": "6 to 8 hours"
        },
        {
            "foodname": "Skillet Braised Chicken",
            "link": "https://medlineplus.gov/recipes/skillet-braised-chicken/",
            "recipe": "25 minutes"
        },
        {
            "foodname": "Skillet Mac and Cheese",
            "link": "https://medlineplus.gov/recipes/skillet-mac-and-cheese/",
            "recipe": "30 minutes"
        },
        {
            "foodname": "Spring Fling Vegetable Chowder",
            "link": "https://medlineplus.gov/recipes/spring-fling-vegetable-chowder/",
            "recipe": "45 minutes"
        },
        {
            "foodname": "Squash Blossom Quesadilla",
            "link": "https://medlineplus.gov/recipes/squash-blossom-quesadilla/",
            "recipe": "25 minutes"
        },
        {
            "foodname": "Stovetop Tuna Casserole",
            "link": "https://medlineplus.gov/recipes/stovetop-tuna-casserole/",
            "recipe": "25 minutes"
        },
        {
            "foodname": "Tuna Veggie Melt",
            "link": "https://medlineplus.gov/recipes/tuna-veggie-melt/",
            "recipe": "20 minutes"
        },
        {
            "foodname": "Turkey Cranberry Quesadilla",
            "link": "https://medlineplus.gov/recipes/turkey-cranberry-quesadilla/",
            "recipe": "10 minutes"
        },
        {
            "foodname": "Turkey Salad",
            "link": "https://medlineplus.gov/recipes/turkey-salad/",
            "recipe": "10 minutes"
        },
        {
            "foodname": "Vegetarian Chili",
            "link": "https://medlineplus.gov/recipes/vegetarian-chili/",
            "recipe": "45 minutes"
        },
        {
            "foodname": "Vegetarian Cowboy Salad",
            "link": "https://medlineplus.gov/recipes/vegetarian-cowboy-salad/",
            "recipe": "20 minutes"
        },
        {
            "foodname": "Vegetarian Tofu Enchiladas",
            "link": "https://medlineplus.gov/recipes/vegetarian-tofu-enchiladas/",
            "recipe": "60 minutes"
        },
        {
            "foodname": "Veggie Quesadillas with Cilantro Yogurt Dip",
            "link": "https://medlineplus.gov/recipes/veggie-quesadillas-with-cilantro-yogurt-dip/",
            "recipe": "35 minutes"
        },
        {
            "foodname": "Veggie Stew",
            "link": "https://medlineplus.gov/recipes/veggie-stew/",
            "recipe": "45 minutes"
        },
        {
            "foodname": "Watermelon and Fruit Salad",
            "link": "https://medlineplus.gov/recipes/watermelon-and-fruit-salad/",
            "recipe": "20 minutes"
        },
        {
            "foodname": "Zucchini Pizza Boats",
            "link": "https://medlineplus.gov/recipes/zucchini-pizza-boats/",
            "recipe": "40 minutes"
        }
    ]
    foodnames = [food["foodname"] for food in foods]
    model_nutritionist = SentenceTransformer('all-MiniLM-L6-v2')


    embeddingsf = model_nutritionist.encode(foodnames)

    dimensionf = embeddingsf.shape[1]  # Assuming embeddings are 2D array
    indexf = faiss.IndexFlatL2(dimensionf)
    indexf.add(embeddingsf)


    def retrievef(query, k=1):
        query_embedding = model_nutritionist.encode([query])
        distances, indices = indexf.search(query_embedding, k)
        return indices[0]
    return output1




