
# Generate query embedding
    query_embedding = model.encode(user_query)

    # Search in vector database
    results = index.query(query_embedding.tolist(), top_k=5, include_metadata=True)
    relevant_chunks = [res['metadata']['chunk'] for res in results['matches']]

    # Use LLM to generate response
    prompt = f"Answer the following question using these chunks: {relevant_chunks}\n\nQuestion: {user_query}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
    # Initialize embedding model and vector database
model = SentenceTransformer('all-MiniLM-L6-v2')
pinecone.init(api_key='your_pinecone_api_key', environment='us-west1-gcp')

# Create Pinecone index
index_name = 'website-embeddings'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)
index = pinecone.Index(index_name)

def crawl_and_scrape(url):
    """Crawl and scrape website content."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])  # Extract paragraphs
    return text

def process_and_store(url):
    """Segment content, generate embeddings, and store in Pinecone."""
    content = crawl_and_scrape(url)
    chunks = [content[i:i+500] for i in range(0, len(content), 500)]  # Segment text
    embeddings = model.encode(chunks)
    
    # Store embeddings in Pinecone
    for i, embedding in enumerate(embeddings):
        metadata = {'url': url, 'chunk': i}
        index.upsert([(f'{url}-{i}', embedding.tolist(), metadat
