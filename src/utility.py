import re

def movie_finder(title, movies):
    # making lower case all the movie titles
    titles = [item.lower() for item in movies['title'].tolist()]
    found = []
    for idx, element in enumerate(titles):
        if re.findall(title.lower(), element):
            found.append(movies['title'][idx])
    return found

#movie_finder(title='separation', movies=movies)