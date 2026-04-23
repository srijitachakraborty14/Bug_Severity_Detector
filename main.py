from src.inference import predict

# Take input from user
text = input("Enter bug description: ")

# Predict severity and solution
severity, solution = predict(text)

# Show results
print("\nPredicted Severity:", severity)
print("Suggested Solution:", solution)