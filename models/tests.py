from model import predict_prediction, load_model
import torch

test_cases = [
    # Appropriate language examples
    "Thank you for your help today.",
    "I hope you have a wonderful day.",
    "Great job on the project, well done!",
    "Let's meet for lunch tomorrow.",
    "Your insights were very valuable in the meeting.",
    "I'm looking forward to collaborating with you.",
    "Please let me know if you need any further assistance.",
    "The presentation was excellent and informative.",
    "I appreciate the opportunity to work with you.",
    "Have a nice weekend!",
    "It was a pleasure meeting you.",
    "Keep up the great work!",
    "Thank you for your guidance and support.",
    "I would be happy to help you.",
    "The product you recommended worked really well.",
    "Please reach out to me anytime.",
    "You did an amazing job with the report.",
    "Your suggestions were very helpful.",
    "Congratulations on the promotion!",
    "Looking forward to our next meeting.",

    # Inappropriate language examples
    "This is a damn waste of time and resources.",
    "Who the hell thought this was a good idea?",
    "Absolute crap; can't believe this made it past the first review.",
    "Just fucking kill yourself",
    "I will send you to hell",
    "I will fuck you and make you naked in front of the entire crowd",
    "Suck my ass",
    "Smell my ass",
    "Suck my dick",
    "An absolute trainwreck. Are you a dunderhead",
    "Shut the fuck up",
    "Shut the fuck up whore",
    "You are just a fucking body artist",
    "You are a bullshit",
    "I wouldn’t wish this crap on my worst enemy.",
    "Suck my dick, you loser",
    "Who on earth came up with this crap? It’s embarrassing.",
    "I will suck your pu**y",
    "You are a dickhead, you bastard",
    "You are a bastard, you fucking naked whore"
]

# Expected outputs (0 for appropriate, 1 for inappropriate)
expected_outputs = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Appropriate language (0)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   # Inappropriate language (1)
]


test_model, tokenizer = load_model()
def run_tests(model, tokenizer, test_cases, expected_outputs):
    passed_tests = 0  # Variable to count passed tests
    for i, (test_input, expected_output) in enumerate(zip(test_cases, expected_outputs), 1):
        # Tokenize and encode the input
        inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)

        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_output = torch.argmax(logits, dim=1).item()  # Get the predicted class (0 or 1)

        # Check if prediction matches the expected output
        if predicted_output == expected_output:
            result = f"TEST {i} Passed ✔️"
            passed_tests += 1
        else:
            result = f"TEST {i} Failed ❌"

        print(result)

    return passed_tests  # Return the number of passed tests

# Run tests and capture the output
passed = run_tests(test_model, tokenizer, test_cases, expected_outputs)

# Compare the number of passed tests to 40
if passed > 0.9*len(test_cases):
    print("PRODUCTION PASSED ✔️. SYSTEM CAN BE DEPLOYED")
else:
    print("PRODUCTION FAILED ❌. SYSTEM NEEDS IMPROVEMENTS")
