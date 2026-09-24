import math
from encoder  import create_embedding
def dot_maker(a , b):
    first_var = 0
    
    for d in range(len(a)):
       x = a[d] * b[d]
       first_var += x

    return (first_var)

def vector_length(a):
    Sqtotal = 0
    for i in a:
        x = i*i
        Sqtotal += x
    length = math.sqrt(Sqtotal)
    return(length)

def cosine_similarity(a, b):
    dot = dot_maker(a, b)
    alength = vector_length(a)
    blength = vector_length(b)

    return dot / (alength * blength)
        




#this def compares the faces
def recognize(unknown_img, data_base):
    unknown_face = create_embedding(unknown_img)
    highest_rate = 0
    best_match = str()
    for person in data_base:
        face = person.get('embedding')
        name = person.get('name')
        score = cosine_similarity(face, unknown_face)    
        if score > highest_rate :
            highest_rate = score
            best_match = name

    if highest_rate >= 0.50:
        return (best_match, highest_rate)
    else: 
        return('Unknown', highest_rate)    