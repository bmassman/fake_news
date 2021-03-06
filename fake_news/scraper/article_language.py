from nltk import wordpunct_tokenize
from nltk.corpus import stopwords


def calculate_languages_ratios(text):
    """
    Compute per language included in nltk number of unique stopwords appearing
    in analyzed text.
    """
    languages_ratios = {}
    tokens = wordpunct_tokenize(text)
    words = {word.lower() for word in tokens}
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        common_elements = words & stopwords_set
        languages_ratios[language] = len(common_elements)
    return languages_ratios


def detect_language(text):
    ratios = calculate_languages_ratios(text)
    most_rated_language = max(ratios, key=ratios.get)
    return most_rated_language


if __name__ == '__main__':

    text = """
    Les deux suspects de 30 et 43 ans, habitant la région parisienne et proches
    de l’assaillant, ont un lourd casier judiciaire, mais sans connotation 
    terroriste, toujours selon cette source, qui précise que, d’après les 
    premières investigations, ils n’ont pas de lien avec la mouvance islamiste 
    radicale. Le plus jeune a également été mis en examen pour détention d’arme 
    de catégorie B, en relation avec une entreprise terroriste. Ils ont tous 
    les deux été placés en détention provisoire, conformément aux réquisitions 
    du parquet de Paris.
    """
    language = detect_language(text)
    print(language)
    text = 'This is some English stuff right here. I hope that it knows.'
    print(detect_language(text))
