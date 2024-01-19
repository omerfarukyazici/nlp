import numpy as np
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
from gensim.models import Word2Vec
#%%
data=pd.read_csv("UpdateData5.csv")
print(data.head())
#%%
all_sentences=''.join(data['Metin'].astype(str))
#%%
words=word_tokenize(all_sentences.lower())

word_frequencies=Counter(words)

top_20_words = word_frequencies.most_common(20)

for word, frequency in top_20_words:
    print(word, frequency)

    #%%
word_frequencies =Counter(words)

# Word2Vec modelini eğit
model = Word2Vec(sentences=[words], vector_size=100, window=5, min_count=1, workers=10, epochs=40)

# Modeli kaydet (isteğe bağlı)
model.save("word2vec_model.model")

# Her kelime için benzer kelimeleri yazdır
for target_word, _ in word_frequencies.most_common(20):
    similar_words = model.wv.most_similar(target_word, topn=5)
    
    print(f"\nBenzer kelimeler for '{target_word}':")
    for word, score in similar_words:
        print(word, score)
#%%        
print("###################################################")

def find_most_similar_sentences(target_sentence, all_sentences, num_similar=4):
    # Cümleleri kelimelere ayır
    target_words = word_tokenize(target_sentence.lower())
    
    # Hedef cümlenin kelime vektörlerini al
    target_vectors = [model.wv[word] for word in target_words if word in model.wv]

    # Eğer hedef cümledeki hiçbir kelime modelde yoksa, benzerlik hesaplamamız mümkün değil
    if not target_vectors:
        return None

    # Tüm cümlelerle benzerlik skorlarını hesapla
    similarity_scores = []
    for sentence in all_sentences:
        words = word_tokenize(sentence.lower())
        vectors = [model.wv[word] for word in words if word in model.wv]
        
        if vectors:
            # Kelime vektörlerinin ortalamasını al
            target_vector_mean = np.mean(target_vectors, axis=0)
            vector_mean = np.mean(vectors, axis=0)

            # Ortalama vektörler arasındaki benzerliği hesapla
            similarity_score = np.dot(target_vector_mean, vector_mean) / (np.linalg.norm(target_vector_mean) * np.linalg.norm(vector_mean))
            similarity_scores.append((sentence, similarity_score))

    # En yüksek benzerlik skoruna sahip cümleleri bul
    similar_sentences = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:num_similar]

    return similar_sentences

# Örnek bir cümle
example_sentence = "film phenomen mani reason first importantli visual impress film seen day anim style alon mystic wonder qualiti anim stop howev ad mani littl fun detail made feel like watch sort blend comic movi addit gorgeou visual natur spider man impress humor meta refer refer cringey spider man scene post credit meme refer person favorit ton littl detail sure never truli appreci never consum comic book imagin film even special marvel comic fan gripe spider man fail keep attent act longer would benefit greatli cut minut said best movi seen month hope review reach someon go see urg reconsid"
# Tüm cümleler
all_data_sentences = data['Metin'].astype(str).tolist()

# En benzer 3 cümleyi bul
similar_sentences = find_most_similar_sentences(example_sentence, all_data_sentences, num_similar=4)

# Sonuçları yazdır
print("Seçili Cümle:", example_sentence)
print("\nEn Benzer 3 Cümle:")
for sentence, similarity_score in similar_sentences:
    index = all_data_sentences.index(sentence)
    print(f"{index}: {sentence} - Benzerlik Skoru: {similarity_score}")
print("\n" * 3)

#%%

print("###################################################")

def find_most_similar_sentences(target_sentence, all_sentences, num_similar=4):
    # Cümleleri kelimelere ayır
    target_words = word_tokenize(target_sentence.lower())
    
    # Hedef cümlenin kelime vektörlerini al
    target_vectors = [model.wv[word] for word in target_words if word in model.wv]

    # Eğer hedef cümledeki hiçbir kelime modelde yoksa, benzerlik hesaplamamız mümkün değil
    if not target_vectors:
        return None

    # Tüm cümlelerle benzerlik skorlarını hesapla
    similarity_scores = []
    for sentence in all_sentences:
        words = word_tokenize(sentence.lower())
        vectors = [model.wv[word] for word in words if word in model.wv]
        
        if vectors:
            # Kelime vektörlerinin ortalamasını al
            target_vector_mean = np.mean(target_vectors, axis=0)
            vector_mean = np.mean(vectors, axis=0)

            # Ortalama vektörler arasındaki benzerliği hesapla
            similarity_score = np.dot(target_vector_mean, vector_mean) / (np.linalg.norm(target_vector_mean) * np.linalg.norm(vector_mean))
            similarity_scores.append((sentence, similarity_score))

    # En yüksek benzerlik skoruna sahip cümleleri bul
    similar_sentences = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:num_similar]

    return similar_sentences

# Örnek bir cümle
example_sentence = "cannot stress enough good film critic actual got thing right visual stun great storylin featur comedi tragedi"
# Tüm cümleler
all_data_sentences = data['Metin'].astype(str).tolist()

# En benzer 3 cümleyi bul
similar_sentences = find_most_similar_sentences(example_sentence, all_data_sentences, num_similar=3)

# Sonuçları yazdır
print("Seçili Cümle:", example_sentence)
print("\nEn Benzer 3 Cümle:")
for sentence, similarity_score in similar_sentences:
    index = all_data_sentences.index(sentence)
    print(f"{index}: {sentence} - Benzerlik Skoru: {similarity_score}")
print("\n" * 3)
#%%

print("###################################################")

def find_most_similar_sentences(target_sentence, all_sentences, num_similar=4):
    # Cümleleri kelimelere ayır
    target_words = word_tokenize(target_sentence.lower())
    
    # Hedef cümlenin kelime vektörlerini al
    target_vectors = [model.wv[word] for word in target_words if word in model.wv]

    # Eğer hedef cümledeki hiçbir kelime modelde yoksa, benzerlik hesaplamamız mümkün değil
    if not target_vectors:
        return None

    # Tüm cümlelerle benzerlik skorlarını hesapla
    similarity_scores = []
    for sentence in all_sentences:
        words = word_tokenize(sentence.lower())
        vectors = [model.wv[word] for word in words if word in model.wv]
        
        if vectors:
            # Kelime vektörlerinin ortalamasını al
            target_vector_mean = np.mean(target_vectors, axis=0)
            vector_mean = np.mean(vectors, axis=0)

            # Ortalama vektörler arasındaki benzerliği hesapla
            similarity_score = np.dot(target_vector_mean, vector_mean) / (np.linalg.norm(target_vector_mean) * np.linalg.norm(vector_mean))
            similarity_scores.append((sentence, similarity_score))

    # En yüksek benzerlik skoruna sahip cümleleri bul
    similar_sentences = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:num_similar]

    return similar_sentences

# Örnek bir cümle
example_sentence = "lush astound stylis artwork meticul blend screen print style graphic edgi colour super funni sweet w almost hammi deadpool type humour fantast movi realli enjoy visual treat sure"
# Tüm cümleler
all_data_sentences = data['Metin'].astype(str).tolist()

# En benzer 3 cümleyi bul
similar_sentences = find_most_similar_sentences(example_sentence, all_data_sentences, num_similar=4)

# Sonuçları yazdır
print("Seçili Cümle:", example_sentence)
print("\nEn Benzer 3 Cümle:")
for sentence, similarity_score in similar_sentences:
    index = all_data_sentences.index(sentence)
    print(f"{index}: {sentence} - Benzerlik Skoru: {similarity_score}")
print("\n" * 3)