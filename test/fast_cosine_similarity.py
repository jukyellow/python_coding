#https://wikidocs.net/24603
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

doc1 = np.array([0,1,1,1])
doc2 = np.array([1,0,1,1])
doc3 = np.array([2,0,2,2])

print(cos_sim(doc1, doc2)) #문서1과 문서2의 코사인 유사도
print(cos_sim(doc1, doc3)) #문서1과 문서3의 코사인 유사도
print(cos_sim(doc2, doc3)) #문서2과 문서3의 코사인 유사도

# tot = np.concatenate((doc1,doc2), axis=0)
# tot = np.concatenate((tot,doc1), axis=0)
# print(tot.shape)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

example = [
    "apple iphone case",
    "apple iphone case",
    "apple iphone acc",
    "apple corperation acc",
    "apple corperation",
    "apple iphone"
]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(example)
print(tfidf_matrix.shape)
print(tfidf_matrix)

tf_vec1 = tfidf.transform([example[0]])
tf_vec2 = tfidf.transform([example[1]])
tf_vec3 = tfidf.transform([example[2]])
tf_vec4 = tfidf.transform([example[3]])

tf_vec5 = tfidf.transform([example[4]])
tf_vec6 = tfidf.transform([example[5]])

cosine_sim = cosine_similarity(tf_vec1, tf_vec2)
print('cosine_sim 1 2:', cosine_sim )
cosine_sim = cosine_similarity(tf_vec1, tf_vec3)
print('cosine_sim 1 3:', cosine_sim )
cosine_sim = cosine_similarity(tf_vec1, tf_vec4)
print('cosine_sim 1 4:', cosine_sim )

cosine_sim = cosine_similarity(tf_vec4, tf_vec5)
print('cosine_sim 4 5 :', cosine_sim )

print('tf_vec4:', tf_vec4.shape)
print('tf_vec5:', tf_vec5.shape)

cosine_sim = cosine_similarity(tf_vec4, tfidf_matrix)
print('cosine_sim 4 ALL :', cosine_sim )
top_result =  (lambda x: x>0.8)(cosine_sim[0])
print('top_result :', top_result )
for (score, text) in zip(top_result, example):
    if score==True:
        print('top:', text)

tf_vec4_5 = tfidf.transform([example[4], example[5]])
print('tf_vec4_5:', tf_vec4_5.shape)
cosine_sim_list = cosine_similarity(tf_vec4_5, tfidf_matrix)
print('cosine_sim_list:', cosine_sim_list.shape)
print('cosine_sim 4 ALL :', cosine_sim_list )
for cosine_sim in cosine_sim_list:
    top_result =  (lambda x: x>0.8)(cosine_sim)
    print('top_result :', top_result )
    for (score, text) in zip(top_result, example):
        if score==True:
            print('top:', text)

# print(list(filter(lambda x: x < 5, range(10))))
# top_result =  list(map(lambda x: x>0.8, cosine_sim[0]))
# print('top_result :', top_result )

# print('cos_sim:', cos_sim(tf_vec4, tf_vec5.T))

# #https://medium.com/analytics-vidhya/speed-up-cosine-similarity-computations-in-python-using-numba-c04bc0741750
# import numpy as np
# from numba import jit
# @jit(nopython=True)
# def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
#     assert(u.shape[0] == v.shape[0])
#     uv = 0
#     uu = 0
#     vv = 0
#     for i in range(u.shape[0]):
#         uv += u[i]*v[i]
#         uu += u[i]*u[i]
#         vv += v[i]*v[i]
#     cos_theta = 1
#     if uu!=0 and vv!=0:
#         cos_theta = uv/np.sqrt(uu*vv)
#     return cos_theta
#
# print('tf_vec4.todense():', tf_vec4.todense())
# print('tf_vec4.toarray():', tf_vec4.toarray())
# print('tf_vec5.toarray():', tf_vec5.toarray())
# print(cosine_similarity_numba(tf_vec4.todense(), tf_vec5.todense()))

# import math
# def norm(vector):
#     return math.sqrt(sum(x * x for x in vector))
# def cosine_similarity_t(vec_a, vec_b):
#         norm_a = norm(vec_a)
#         norm_b = norm(vec_b)
#         dot = sum(a * b for a, b in zip(vec_a, vec_b))
#         return dot / (norm_a * norm_b)
# print(cosine_similarity_t(tf_vec4, tf_vec5))

# # Define a function to calculate the cosine similarities a few different ways
# def calc_sim(A, B):
#     similarity = np.dot(A, B.T)
#     print('similarity:', similarity.shape, similarity)
#     # squared magnitude of preference vectors (number of occurrences)
#     square_mag = np.diag(similarity)
#     # inverse squared magnitude
#     inv_square_mag = 1 / square_mag
#     # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
#     inv_square_mag[np.isinf(inv_square_mag)] = 0
#     # inverse of the magnitude
#     inv_mag = np.sqrt(inv_square_mag)
#     # cosine similarity (elementwise multiply by inverse magnitudes)
#     cosine = similarity * inv_mag
#     return cosine.T * inv_mag
# print(calc_sim(tf_vec4, tf_vec5))


# import sklearn.preprocessing as pp
# def cosine_similarities(mat):
#     col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
#     return col_normed_mat.T * col_normed_mat
# def cosine_similaritie2(mat1, mat2):
#     col_normed_mat1 = pp.normalize(mat1.tocsc(), axis=0)
#     col_normed_mat2 = pp.normalize(mat2.tocsc(), axis=0)
#     return col_normed_mat1 * col_normed_mat2
#
# cosine_sim = cosine_similarities(tf_vec4)
# print('cosine_sim :', cosine_sim )
# cosine_sim = cosine_similaritie2(tf_vec4, tf_vec5)
# print('cosine_sim :', cosine_sim )


from fastdist import fastdist
import numpy as np
from scipy.spatial import distance

# #a, b = np.random.rand(200, 100), np.random.rand(2500, 100)
# #%timeit -n 100
# print(fastdist.matrix_to_matrix_distance(tf_vec4, tf_vec5, fastdist.cosine, "cosine"))
# # 8.97 ms Â± 11.2 ms per loop (mean Â± std. dev. of 7 runs, 100 loops each)
# note this high stdev is because of the first run taking longer to compile

# #%timeit -n 100
# print(distance.cdist(tf_vec4, tf_vec5, "cosine"))
# # 57.9 ms Â± 4.43 ms per loop (mean Â± std. dev. of 7 runs, 100 loops each)