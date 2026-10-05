import xml.etree.ElementTree as ET
import os
import re
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
import math

def read_stopwords(file_path):
    stopwords_set = set()
    try:
        with open(file_path, 'r') as file:
            for line in file:
                stopwords_set.add(line.strip().lower())
    except FileNotFoundError:
        print("Stopwords file not found.")
    return stopwords_set

def parse_xml_text(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        text_data = []

        title_elem = root.find("title")
        if title_elem is not None and title_elem.text:
            text_data.append(title_elem.text.strip())

        text_elem = root.find("text")
        if text_elem is not None:
            for paragraph in text_elem.findall("p"):
                if paragraph.text:
                    text_data.append(paragraph.text.strip())

        return text_data
    except:
        return []

def extract_label_from_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        metadata = root.find("metadata")
        if metadata is not None:
            for codes_group in metadata.findall("codes"):
                if codes_group.attrib.get("class") == "bip:topics:1.0":
                    first_code = codes_group.find("code")
                    if first_code is not None:
                        return first_code.attrib.get("code", "unknown")
        return "unknown"
    except:
        return "unknown"


def filter_and_stem_text(text_data, stopwords_set, stemmer):
    unique_stems = set()
    for text in text_data:
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\b\d+(\.\d+)?\b', '', text)
        words = text.split()
        for word in words:
            word = word.lower()
            if word not in stopwords_set and word.isalpha():
                stemmed_word = stemmer.stem(word)
                unique_stems.add(stemmed_word)
    return list(unique_stems)

def generate_global_vector(stemmed_text, global_vector, start_index):
    index = start_index
    for word in stemmed_text:
        if word not in global_vector:
            global_vector[word] = index
            index += 1
    return global_vector, index

def generate_rar_vector(global_vector, stemmed_text):
    word_to_index = global_vector
    rar_vector = Counter()
    for word in stemmed_text:
        if word in word_to_index:
            rar_vector[word_to_index[word]] += 1
    return rar_vector

def calculate_entropy(label_counts):
    total = sum(label_counts.values())
    if total == 0:
        return 0
    entropy = 0.0
    for count in label_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def calculate_information_gain(global_vector, documents_rar, labels):
    total_entropy = calculate_entropy(Counter(labels))
    feature_gain = {}

    for word, index in global_vector.items():
        with_feature = []
        without_feature = []
        for i, rar in enumerate(documents_rar):
            if index in rar:
                with_feature.append(labels[i])
            else:
                without_feature.append(labels[i])
        entropy_with = calculate_entropy(Counter(with_feature))
        entropy_without = calculate_entropy(Counter(without_feature))
        total = len(labels)
        weighted_entropy = (len(with_feature)/total) * entropy_with + (len(without_feature)/total) * entropy_without
        gain = total_entropy - weighted_entropy
        feature_gain[word] = gain

    return feature_gain

def create_labels_file(directory, output_file):
    labels = []
    unique_labels = set()
    with open(output_file, 'w') as f:
        for file in sorted(os.listdir(directory)):
            if file.lower().endswith('.xml'):
                file_path = os.path.join(directory, file)
                label = extract_label_from_xml(file_path)
                labels.append(label)
                unique_labels.add(label)
        
        for label in sorted(unique_labels):
            f.write(label + '\n')
    
    return labels


def vectorize_query(query, stopwords_set, stemmer, global_vector):
    query_stemmed = filter_and_stem_text([query], stopwords_set, stemmer)
    query_vector = [0] * len(global_vector)
    for word in query_stemmed:
        if word in global_vector:
            index = global_vector[word]
            query_vector[index] += 1
    return query_vector

def vector_to_list(rar_vector, length):
    full_vector = [0] * length
    for idx, count in rar_vector.items():
        full_vector[idx] = count
    return full_vector

def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def euclidean_distance(vec1, vec2):
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    return 1 / (1 + dist) 

def manhattan_distance(vec1, vec2):
    dist = sum(abs(a - b) for a, b in zip(vec1, vec2))
    return 1 / (1 + dist) 

def idf_vector(vector, doc_index):
    tf_idf = []
    num_docs = len(all_rar_vectors)

    for i, tf in enumerate(vector):
        df = sum(1 for doc in all_rar_vectors if i in doc)
        idf = math.log((num_docs + 1) / (df + 1)) + 1 
        tf_idf.append(tf * idf)
    return tf_idf


def normalize_vector(vector, method, doc_index=None):
    if method == 'binary':
        return [1 if v > 0 else 0 for v in vector]

    elif method == 'log':
        return [1 + math.log(1 + math.log(v)) if v > 0 else 0 for v in vector]

    elif method == 'nominal':
        max_val = max(vector)
        if max_val == 0:
            return [0 for v in vector]
        return [v / max_val for v in vector]

    elif method == 'sum1':
        total = sum(vector)
        if total == 0:
            return [0 for v in vector]
        return [v / total for v in vector]

    elif method == 'idf':
        return idf_vector(vector, doc_index)

    else:
        raise ValueError("Unknown normalization method")



if __name__ == "__main__":
    directory = "Reuters\\Reuters_34"
    stopwords_file = "stopwords.txt"
    labels_file = "labels.txt"

    stopwords_set = read_stopwords(stopwords_file)
    global_vector = {}
    start_index = 0
    all_rar_vectors = []
    all_labels = []

    stemmer = PorterStemmer()

    all_files = sorted(os.listdir(directory))
    for file in all_files:
        if not file.lower().endswith('.xml'):
            continue
        file_path = os.path.join(directory, file)
        xml_text = parse_xml_text(file_path)
        stemmed_text = filter_and_stem_text(xml_text, stopwords_set, stemmer)
        global_vector, start_index = generate_global_vector(stemmed_text, global_vector, start_index)
        rar_vector = generate_rar_vector(global_vector, stemmed_text)
        all_rar_vectors.append(rar_vector)

    all_labels = create_labels_file(directory, labels_file)

    info_gain = calculate_information_gain(global_vector, all_rar_vectors, all_labels)

    threshold = 0.1

    sorted_gain = sorted(info_gain.items(), key=lambda x: x[1], reverse=True)

    filtered_gain = [(word, gain) for word, gain in sorted_gain if gain >= threshold]

    print("\nTop 10 features by Information Gain (with threshold >= 0.5):")
    for word, gain in filtered_gain:
        print(f"{word}: {gain:.4f}")

    print("\nChoose the normalization method:")
    print("  - binary\n  - log\n  - nominal\n  - sum1\n  - idf")
    norm_method = input("Normalization method ").strip().lower()

    print("\nChoose the similarity method:")
    print("  - cosine\n  - euclidean\n  - manhattan")
    sim_method = input("Similarity method ").strip().lower()


    query = input("\nEnter your query: ")
    query_vector_raw = vectorize_query(query, stopwords_set, stemmer, global_vector)
    query_vector = normalize_vector(query_vector_raw, norm_method)

    document_vectors_raw = [vector_to_list(rar, len(global_vector)) for rar in all_rar_vectors]

    similarities = []
    for i, doc_vector_raw in enumerate(document_vectors_raw):
        doc_vector = normalize_vector(doc_vector_raw, norm_method, i)

        if sim_method == 'cosine':
            sim = cosine_similarity(query_vector, doc_vector)
        elif sim_method == 'euclidean':
            sim = euclidean_distance(query_vector, doc_vector)
        elif sim_method == 'manhattan':
            sim = manhattan_distance(query_vector, doc_vector)
        else:
            print("Unknown similarity method")
            sim = 0

        similarities.append((all_files[i], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\nSimilar documents with query  (threshold >= {threshold}):")
    for filename, sim in similarities:
        if sim >= threshold:
            print(f"{filename}: similarity = {sim:.4f}")
