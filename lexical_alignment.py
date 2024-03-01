import stanza
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from extract_chunks import extract_chunks
import json

def convert_conll_list_to_string(doc):
    try:
        conll_string = ""
        for sentence in doc.sentences:
            for token in sentence.tokens:
                for word in token.words:
                    line = "\t".join([
                        str(word.id) if word.id is not None else '_',
                        word.text if word.text is not None else '_',
                        word.lemma if word.lemma is not None else '_',
                        word.upos if word.upos is not None else '_',
                        word.xpos if word.xpos is not None else '_',
                        word.feats if word.feats is not None else '_',
                        str(word.head) if word.head is not None else '0',
                        word.deprel if word.deprel is not None else '_',
                        word.deps if word.deps is not None else '_',
                        word.misc if word.misc is not None else '_'
                    ])
                    conll_string += line + '\n'
            conll_string += '\n'
        return conll_string
    except Exception as e:
        print(f"An error occurred in document_to_conll_string: {e}")
        return ""


def align_lexical(l1, l2, x, y, encoder, sents1, sents2, file_name):
    try:
        # Load Stanza models for the specified languages
        nlp_l1 = stanza.Pipeline(lang=l1, processors='tokenize,mwt,pos,lemma,depparse')
        nlp_l2 = stanza.Pipeline(lang=l2, processors='tokenize,mwt,pos,lemma,depparse')

        # Initialize the list to store final word or chunk alignments
        alignments = []

        # Iterate over each group of aligned sentences
        # The function zip(x, y) pairs each element of x with the corresponding element in y,
        # allowing the loop to process these pairs in tandem
        # La fonction zip(x, y) associe chaque élément de x avec l'élément correspondant dans y,
        # permettant à la boucle de traiter ces paires en tandem
        for group_x, group_y in zip(x, y):
            # Concatenate sentences in each group to form a single text for parsing
            text_l1 = ' '.join([sents1[i-1] for i in group_x])  # Adjust indices for 0-based indexing
            text_l2 = ' '.join([sents2[i-1] for i in group_y])

            # Process texts with Stanza to get CoNLL-U formatted data
            doc_l1 = nlp_l1(text_l1)
            doc_l2 = nlp_l2(text_l2)

            # Use the function to convert documents to CoNLL format
            conll_l1 = convert_conll_list_to_string(doc_l1)
            conll_l2 = convert_conll_list_to_string(doc_l2)

            # Now, you can extract chunks from the CoNLL data
            chunks_l1 = extract_chunks(conll_l1)
            chunks_l2 = extract_chunks(conll_l2)

            if not chunks_l1 or not chunks_l2:
                print("One of the languages has no chunks, skipping similarity calculation for this pair.")
                continue
            # Compute embeddings for each chunk
            chunk_embeds_l1 = encoder.encode([chunk[0] for chunk in chunks_l1])
            chunk_embeds_l2 = encoder.encode([chunk[0] for chunk in chunks_l2])

            if chunk_embeds_l1.ndim == 1:
                chunk_embeds_l1 = chunk_embeds_l1.reshape(1, -1)
            if chunk_embeds_l2.ndim == 1:
                chunk_embeds_l2 = chunk_embeds_l2.reshape(1, -1)

            # Find the best matching chunks based on cosine similarity
            similarity_matrix = cosine_similarity(chunk_embeds_l1, chunk_embeds_l2)

            # For each chunk in l1, find the best matching chunk in l2
            for i, row in enumerate(similarity_matrix):
                best_match_index = np.argmax(row)
                best_match_score = float(row[best_match_index])
                alignments.append({
                    'l1_chunk': chunks_l1[i][0],
                    'l2_chunk': chunks_l2[best_match_index][0],
                    'similarity': best_match_score
                })

        # print top 10 lexical alignments
        for alignment in alignments[:10]:
            print(f"Chunk in {l1}: {alignment['l1_chunk']}")
            print(f"Chunk in {l2}: {alignment['l2_chunk']}")
            print(f"Similarity score: {alignment['similarity']:.2f}")
            print("-" * 30)

        # File name for the raw data
        raw_file_name = file_name + '_aligned_chunks.json'

        # Dumping the raw data
        with open(raw_file_name, 'w', encoding='utf-8') as file:
            json.dump(alignments, file, ensure_ascii=False, indent=4)

        return alignments
    except Exception as e:
        print(f"An error occurred in align_lexical: {e}")
        return []