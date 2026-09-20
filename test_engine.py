from core.rag_engine import ask_question

# RAG engine ko test karna
question = "What is RAG?"

answer, timestamp = ask_question(question)

print("\nANSWER:")
print(answer)

print("\nTIMESTAMP:")
print(timestamp)