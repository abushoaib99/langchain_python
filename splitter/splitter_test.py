from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter

# Example text
text = """LangChain makes it easier to build applications with LLMs.
It provides prompt management, chains, memory, and integrations.
RecursiveCharacterTextSplitter is a smart way to chunk text into pieces."""

# 1. Using RecursiveCharacterTextSplitter (practical splitter)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

recursive_chunks = recursive_splitter.split_text(text)
print("RecursiveCharacterTextSplitter output:\n")
for i, chunk in enumerate(recursive_chunks, 1):
    print(f"{i}: {chunk}")

# 2. Using TextSplitter directly (base class - not smart)
# Normally you shouldn't use this directly, but here’s how:
class SimpleSplitter(TextSplitter):
    def split_text(self, text: str):
        # Just naive fixed-length chunks
        return [text[i:i+50] for i in range(0, len(text), 50)]

simple_splitter = SimpleSplitter(chunk_size=50, chunk_overlap=10)
simple_chunks = simple_splitter.split_text(text)

print("\n\n\nTextSplitter (naive) output:\n")
for i, chunk in enumerate(simple_chunks, 1):
    print(f"{i}: {chunk}")
