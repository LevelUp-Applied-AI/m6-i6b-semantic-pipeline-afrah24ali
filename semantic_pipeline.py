"""
Module 6 Week B — Integration: NER + Embeddings Semantic Pipeline"""
import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.metrics.pairwise import cosine_similarity



def load_and_preprocess(filepath):
    """Load the climate articles dataset and prepare texts for processing."""
    df = pd.read_csv(filepath)
    if"text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column")
    df = df.dropna(subset=['text'])

    df['text']=df['text'].astype(str).str.strip()
    df=df[df['text']!=""]
    if 'language' in df.columns:
        df = df[df['language']=='en']

        df =df.reset_index(drop=True)
    return df 


def run_ner(texts):
    """Run named entity recognition on a list of texts using spaCy.
    """
    nlp=spacy.load("en_core_web_sm")
    entities=[]
    for idx,text in enumerate(texts):
        doc =nlp(text)
        for ent in doc.ents:
            entities.append({
                "text_index":idx,
                "entity_text":ent.text,
                "entity_label":ent.label_
            })
    return pd.DataFrame(entities)

def compute_embeddings(texts, tokenizer, model):
    """Compute DistilBERT embeddings for a list of texts."""
    
    embeddings =[]
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors ="pt",
            truncation = True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs=model(**inputs)
        last_hidden_state=outputs.last_hidden_state
        embedding =last_hidden_state.mean(dim=1)
        embeddings.append(embedding.squeeze().numpy())
    return np.array(embeddings)        


def semantic_search(query, corpus_embeddings, corpus_texts, top_k=5):
    """Find the top-k most similar texts to the query using cosine similarity.
    """
    query = query.reshape(1,-1)
    similarities =cosine_similarity(query,corpus_embeddings)[0]
    top_indices =np.argsort(similarities)[::-1][:top_k]
    results =[]
    for idx in top_indices:
        results.append(
            (corpus_texts[idx],similarities[idx])
        )
    return results    

def enrich_with_entities(search_results, entity_df, corpus_texts):
    """Enrich semantic search results with NER entities."""
    enriched_results =[]
    for text , score in search_results:
        text_index =corpus_texts.index(text)
        matched_entities =entity_df[entity_df["text_index"]== text_index
                                     ]
        entities=[]
        for _, row in matched_entities.iterrows():
            entities.append({
                "text":row["entity_text"],
                "label":row["entity_label"]
            })

        enriched_results.append({
            "text":text,
            "similarity":score,
            "entities":entities
        })    
    return enriched_results    

def demonstrate_pipeline(corpus_df, entity_df, embeddings, queries,
                         tokenizer, model):
    """Run the full pipeline demonstration on example queries.
    """
    results_dict ={}
    corpus_texts=corpus_df["text"].tolist()
    for query in queries:
        query_emb= compute_embeddings([query],tokenizer,model)[0]
        search_results =semantic_search(
            query_emb,
            embeddings,
            corpus_texts

        )
        enriched=enrich_with_entities(
            search_results,
            entity_df,
            corpus_texts
        )
        results_dict[query]= enriched
    return results_dict



if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    # Load and preprocess
    df = load_and_preprocess("data/climate_articles.csv")
    if df is not None:
        texts = df["text"].tolist()
        print(f"Loaded {len(texts)} texts")

        # NER
        entities = run_ner(texts)
        if entities is not None:
            print(f"Extracted {len(entities)} entities")

        # Embeddings
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        model.eval()
        embs = compute_embeddings(texts, tokenizer, model)
        if embs is not None:
            print(f"Embedding matrix shape: {embs.shape}")

        # Demo queries
        with open("data/example_queries.txt") as f:
            queries = [line.strip() for line in f if line.strip()]

        if embs is not None and entities is not None:
            results = demonstrate_pipeline(
                df, entities, embs, queries, tokenizer, model
            )
            if results:
                for q, enriched in results.items():
                    print(f"\nQuery: {q}")
                    for r in enriched[:3]:
                        print(f"  Score: {r['similarity']:.4f}")
                        print(f"  Text: {r['text'][:100]}...")
                        print(f"  Entities: {r['entities'][:5]}")
