"""Deterministic replies for the scripted parts of the call (intro, ID check). Safe, short, three languages.
Other languages fall back to the LLM with a directive."""
import re

INTRO = {
    "en": "Hi, I'm ZentroMart's AI support assistant.",
    "hi": "नमस्ते, मैं ZentroMart का AI सपोर्ट असिस्टेंट हूँ।",
    "hi-en": "Namaste, main ZentroMart ka AI support assistant hoon.",
}
T = {
    "ask_order_id": {
        "en": "Could you please share your order ID?",
        "hi": "कृपया अपना ऑर्डर आईडी बताइए।",
        "hi-en": "Kya aap apna order ID bata sakte hain?",
    },
    "order_id_unclear": {
        "en": "Sorry, I didn't catch the full order ID. It has eight digits. Could you please say it again?",
        "hi": "क्षमा करें, पूरा ऑर्डर आईडी सुनाई नहीं दिया। यह आठ अंकों का होता है, कृपया दोबारा बताइए।",
        "hi-en": "Sorry, poora order ID samajh nahi aaya. Ye aath digits ka hota hai, kripya dobara bataiye.",
    },
    "ask_verification": {
        "en": "Thank you. To keep your order details safe, please tell me the last four digits of your registered phone number, or the delivery pincode.",
        "hi": "धन्यवाद। आपकी सुरक्षा के लिए कृपया अपने रजिस्टर्ड फ़ोन नंबर के आख़िरी चार अंक या डिलीवरी पिनकोड बताइए।",
        "hi-en": "Thank you. Suraksha ke liye, kripya apne registered phone number ke aakhri 4 digits ya delivery pincode bataiye.",
    },
    "retry_verification": {
        "en": "Sorry, that doesn't match our records. Could you please check and try again with the last four digits of your registered phone number, or the delivery pincode?",
        "hi": "क्षमा करें, यह हमारे रिकॉर्ड से मेल नहीं खाता। कृपया जाँचकर दोबारा रजिस्टर्ड फ़ोन नंबर के आख़िरी चार अंक या डिलीवरी पिनकोड बताइए।",
        "hi-en": "Sorry, ye hamare record se match nahi hua. Kripya check karke dobara registered phone ke aakhri 4 digits ya delivery pincode bataiye.",
    },
    "verification_failed": {
        "en": "I'm sorry, I couldn't verify you, so I can't share any order details. Please contact support from your registered phone number. I'm happy to help with general policy questions.",
        "hi": "क्षमा करें, आपकी पहचान की पुष्टि नहीं हो पाई, इसलिए ऑर्डर की जानकारी साझा नहीं की जा सकती। कृपया अपने रजिस्टर्ड फ़ोन नंबर से सपोर्ट से संपर्क करें। सामान्य नीति के बारे में मदद ज़रूर की जा सकती है।",
        "hi-en": "Sorry, aapki pehchaan verify nahi ho payi, isliye order ki details share nahi ki ja sakti. Kripya apne registered phone number se support se contact kijiye. General policy ke baare mein madad zaroor ki ja sakti hai.",
    },
    "verified_ok": {
        "en": "Thank you, you're verified. How can I help with this order?",
        "hi": "धन्यवाद, आपकी पुष्टि हो गई है। इस ऑर्डर में आपकी क्या मदद की जा सकती है?",
        "hi-en": "Thank you, aapki verification ho gayi hai. Is order mein aapki kya madad ki ja sakti hai?",
    },
    "escalated": {
        "en": "I'm sorry about this. I've created a ticket and a human colleague will contact you shortly. I've passed on a short summary so you won't need to repeat yourself.",
        "hi": "इसके लिए खेद है। मैंने टिकट बना दिया है और हमारे एक सहयोगी जल्द ही आपसे संपर्क करेंगे। मैंने संक्षिप्त सारांश भी दे दिया है, इसलिए आपको दोबारा बताना नहीं पड़ेगा।",
        "hi-en": "Iske liye sorry. Maine ticket bana diya hai aur hamare ek colleague jaldi aapse contact karenge. Maine short summary bhi de di hai, aapko dobara batana nahi padega.",
    },
    "escalation_failed": {
        "en": "I'm sorry, I couldn't create the handover ticket just now. Please try again in a moment, or contact support directly.",
        "hi": "क्षमा करें, अभी टिकट नहीं बन पाया। कृपया थोड़ी देर में दोबारा कोशिश करें या सपोर्ट से सीधे संपर्क करें।",
        "hi-en": "Sorry, abhi ticket nahi ban paya. Kripya thodi der mein dobara try kijiye ya support se seedha contact kijiye.",
    },
    "greeting": {
        "en": "How can I help you today?",
        "hi": "आज आपकी क्या मदद की जा सकती है?",
        "hi-en": "Aaj aapki kya madad ki ja sakti hai?",
    },
    "didnt_catch": {
        "en": "Sorry, I didn't catch that. Could you say it again?",
        "hi": "क्षमा करें, आपकी बात सुनाई नहीं दी। क्या आप दोबारा बोल सकते हैं?",
        "hi-en": "Sorry, aapki baat sunai nahi di. Kya aap dobara bol sakte hain?",
    },
    "llm_down": {
        "en": "I'm sorry, I'm having trouble right now. Could you please try again in a moment?",
        "hi": "क्षमा करें, अभी मुझे कुछ समस्या हो रही है। कृपया थोड़ी देर में दोबारा कोशिश करें।",
        "hi-en": "Sorry, abhi mujhe thodi problem ho rahi hai. Kripya thodi der mein dobara try kijiye.",
    },
}

AI_PATTERN = re.compile(r"\bAI\b|artificial intelligence|एआई|कृत्रिम|ఏఐ|ஏஐ|ಎಐ|എഐ|એઆઈ|এআই", re.I)


def get(kind: str, lang: str):
    """Template text or None when the language has no template."""
    d = T[kind]
    return d.get(lang) or (d.get("hi") if lang == "mr" else None)


def intro(lang: str):
    return INTRO.get(lang)
