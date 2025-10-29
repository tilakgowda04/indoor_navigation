import whisper
import ollama
from langdetect import detect
# Step 1: Transcribe the audio using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("large")  # Use 'base' for speed; 'small' or 'medium' for better accuracy (larger models)
    result = model.transcribe(audio_file, language="hi")
    transcription = result['text']
    print("Transcription:\n - app.py:9", transcription)  # For debugging
    return transcription

# Step 2: Prepare the prompt for LLM based on your parameters, labels, and etiquette
def create_prompt(transcription):
    prompt = f"""
    You are an AI call quality evaluator. Analyze the following call transcription (which may be in Hindi, English, or a mix of both) and classify it into EXACTLY ONE of these categories: Excellent, Good, Satisfactory, Average, Bad. Do NOT use any other category (e.g., 'Mixed'). Even if the transcription has mixed languages or unclear parts, evaluate based on the presence of parameters and adherence to etiquette rules.

    Parameters to check (पैरामीटर जांचें):
    1) Greetings (अभिवादन, जैसे नमस्ते या हैलो)
    2) Delay reason (If more than zero DPD) (देरी का कारण, अगर DPD जीरो से ज्यादा हो)
    3) PENALTY CHARGES (पेनल्टी चार्जेस)
    4) CIBIL AND CREDIT SCORE IMPACT (सिबिल और क्रेडिट स्कोर पर प्रभाव)
    5) FUTURE LOAN REJECTIONS (For more than zero DPD) (भविष्य में लोन रिजेक्शन, अगर DPD जीरो से ज्यादा हो)
    6) PTP (Promise to pay) (वादा भुगतान)
    7) Mode of Payment (भुगतान का तरीका)
    8) Call Back (कॉल बैक)
    9) Zero dpd (Today) cases agents have to inform regarding the EMI amount. (जीरो DPD केस में आज EMI अमाउंट की जानकारी दें)
    10) For More than 5 days dpd cases agents have informed regarding the last 3 broken ptp dates (if there are any broken ptp) (5 दिन से ज्यादा DPD में पिछले 3 ब्रोकन PTP डेट्स की जानकारी दें, अगर हों)

    Excellent calls: On call Payments (कॉल पर पेमेंट), Delightful customer (खुश कस्टमर).
    Good calls: All parameters covered (सभी पैरामीटर कवर), Convinced customer to pay (कस्टमर को पेमेंट के लिए कन्विंस किया), No agent mistakes (एजेंट से कोई गलती नहीं).
    Satisfactory calls: Wrong log creation (गलत लॉग क्रिएशन), Wrong/No delay reason (गलत/कोई देरी कारण नहीं), Missed one parameter (एक पैरामीटर मिस). (6 satisfactory = 1 average)
    Average calls: Missed 2 parameters (2 पैरामीटर मिस), Poor convincing (कन्विंसिंग स्किल कम), Lack of payment tracking (पेमेंट ट्रैकिंग की कमी), Exceeding 6 satisfactory, Unnecessary voice raise (बेवजह आवाज ऊंची), Distraction (डिस्ट्रैक्शन), Extension >3 days (एक्सटेंशन 3 दिन से ज्यादा), Unanswered (अनआंसर्ड), Humiliating/sarcastic (अपमानजनक/व्यंग्यात्मक). (5 average = disqualification)
    Bad calls: Abusive/rude (अपशब्द/रूड), Wrong info (गलत जानकारी), False promises (झूठे वादे), Threatening (धमकी). (1 bad = no incentive)

    Calling Etiquette (Do's and Don'ts) (कॉलिंग एटिकेट):
    - Always introduce yourself and company (हमेशा खुद और कंपनी का परिचय दें).
    - Customer identification (कस्टमर आईडेंटिफिकेशन).
    - Speak clearly, proper info (साफ बोलें, सही जानकारी दें).
    - Follow guidelines (गाइडलाइंस फॉलो करें).
    - Appropriate tone (उचित टोन).
    - Professional (no giggling, noise) (प्रोफेशनल, कोई हंसी या शोर नहीं).
    - Avoid arguing with third parties (थर्ड पार्टी से बहस न करें).
    - Maintain professionalism (प्रोफेशनलизм बनाए रखें).
    - Respect customers (कस्टमर का सम्मान करें).
    - Don't cover phone; use hold/mute (फोन कवर न करें; होल्ड/म्यूट यूज करें).
    - No abusive/sarcastic words (अपशब्द/व्यंग्य न करें).
    - Don't share details without consent (बिना सहमति डिटेल्स शेयर न करें).
    - No fake logs/drops (फेक लॉग/ड्रॉप न करें).
    - No threats (धमकी न दें).
    - No personal meetings (पर्सनल मीटिंग न करें).
    - No false commitments (झूठे कमिटमेंट न दें).
    - No personal numbers/conversations (पर्सनल नंबर/बातचीत न करें).
    - No vague info (अस्पष्ट जानकारी न दें).
    - No unauthorized messages/documents (अनऑथराइज्ड मैसेज/डॉक्यूमेंट न शेयर करें).
    - No adding to social media (सोशल मीडिया में ऐड न करें).
    - No negative company comments (कंपनी के नेगेटिव कमेंट्स न दें).
    - No religious sentiments (धार्मिक भावनाओं को ठेस न पहुंचाएं).
    - No sharing personal bank details (पर्सनल बैंक डिटेल्स शेयर न करें).

    Transcription: {transcription}

    Instructions:
    - Analyze in the context of Hindi if the transcription is in Hindi or mixed Hindi-English.
    - Ignore language mixing when classifying; focus only on parameters and etiquette.
    - If transcription is unclear, prioritize detecting any Bad call criteria (e.g., threats, abusive language) first, then check for missing parameters or etiquette violations.
    - Output EXACTLY in this format: "<Category>\nReason: <1-2 sentences explaining the classification in English>"

    Example Output: 
    Good
    Reason: All parameters were covered, and the agent was professional.
    """
    return prompt

# Step 3: Classify using Ollama LLM
def classify_call(transcription):
    prompt = create_prompt(transcription)
    response = ollama.generate(model='mistral', prompt=prompt)
    classification = response['response'].strip()
    return classification

# Main function to run the process
if __name__ == "__main__":
    audio_file = "hindi_sample_call.wav"  # Replace with your file name
    transcription = transcribe_audio(audio_file)
    result = classify_call(transcription)
    print("\nClassification Result:\n - app.py:86", result)
