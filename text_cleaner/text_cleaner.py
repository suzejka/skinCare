def clear_skin_type_misspelled_data(datasetToClean):
    '''
    Clean misspelled data in skin type column - "Typ cery"
    '''
    datasetToClean.loc[(datasetToClean["Typ cery"] == "tłusta") | 
                        (datasetToClean["Typ cery"] == "tlusta") | 
                        (datasetToClean["Typ cery"] == "Tlusta"), "Typ cery"] = "Tłusta"
    datasetToClean.loc[(datasetToClean["Typ cery"] == "mieszana"), "Typ cery"] = "Mieszana"
    datasetToClean.loc[(datasetToClean["Typ cery"] == "sucha"), "Typ cery"] = "Sucha"
    datasetToClean.loc[(datasetToClean["Typ cery"] == "normalna"), "Typ cery"] = "Normalna"

def clean_nadprodukcja_sebum(datasetToClean, mainOrsecondProblem):
    '''
    Clean misspelled data in main or second problem column - "nadprodukcja sebum"
    '''
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "nadprodukcja sebum") | 
                        (datasetToClean[mainOrsecondProblem] == "nadprodukcja Sebum"), mainOrsecondProblem] = "Nadprodukcja sebum"

def clean_niedoskonalosci(datasetToClean, mainOrsecondProblem):
    '''
    Clean misspelled data in main or second problem column - "niedoskonałości"
    '''
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "niedoskonałości") | 
                        (datasetToClean[mainOrsecondProblem] == "niedoskonalosci") |
                        (datasetToClean[mainOrsecondProblem] == "niedoskonałosci") |
                        (datasetToClean[mainOrsecondProblem] == "niedoskonalości") |
                        (datasetToClean[mainOrsecondProblem] == "Niedoskonalości") | 
                        (datasetToClean[mainOrsecondProblem] == "Niedoskonalosci") |
                        (datasetToClean[mainOrsecondProblem] == "Niedoskonałosci"), mainOrsecondProblem] = "Niedoskonałości"
            
def clean_podraznienie(datasetToClean, mainOrsecondProblem):
    '''
    Clean misspelled data in main or second problem column - "podrażnienie"
    '''
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "podrażnienie") | 
                        (datasetToClean[mainOrsecondProblem] == "podraznienie") |
                        (datasetToClean[mainOrsecondProblem] == "Podraznienie") , mainOrsecondProblem] = "Podrażnienie"
                    
def clean_przebarwienia(datasetToClean, mainOrsecondProblem):
    '''
    Clean misspelled data in main or second problem column - "przebarwienia"
    '''
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "przebarwienia"), mainOrsecondProblem] = "Przebarwienia"

def clean_rozszerzone_pory(datasetToClean, mainOrsecondProblem):
    '''
    Clean misspelled data in main or second problem column - "rozszerzone pory"
    '''
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "rozszerzone pory") | 
                        (datasetToClean[mainOrsecondProblem] == "Rozszerzone Pory") |
                        (datasetToClean[mainOrsecondProblem] == "rozszerzone Pory") , mainOrsecondProblem] = "Rozszerzone pory"

def clean_suche_skorki(datasetToClean, mainOrsecondProblem):
    '''
    Clean misspelled data in main or second problem column - "suche skórki"
    '''
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "suche skórki") | 
                        (datasetToClean[mainOrsecondProblem] == "suche Skórki") |
                        (datasetToClean[mainOrsecondProblem] == "Suche skorki") |
                        (datasetToClean[mainOrsecondProblem] == "suche skorki") |
                        (datasetToClean[mainOrsecondProblem] == "suche Skorki"), mainOrsecondProblem] = "Suche skórki"

def clean_szara_cera(datasetToClean, mainOrsecondProblem):
    '''
    Clean misspelled data in main or second problem column - "szara cera"
    '''
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "szara cera") | 
                        (datasetToClean[mainOrsecondProblem] == "szara Cera") |
                        (datasetToClean[mainOrsecondProblem] == "Szara Cera"), mainOrsecondProblem] = "Szara cera"

def clean_widoczne_naczynka(datasetToClean, mainOrsecondProblem):
    '''
    Clean misspelled data in main or second problem column - "widoczne naczynka"
    '''
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "widoczne naczynka") | 
                        (datasetToClean[mainOrsecondProblem] == "Widoczne Naczynka") |
                        (datasetToClean[mainOrsecondProblem] == "widoczne Naczynka"), mainOrsecondProblem] = "Widoczne naczynka"

def clean_main_or_second_problem_misspelled_data(datasetToClean, mainOrsecondProblem):
    '''
    Clean misspelled data in main or second problem column
    '''
    clean_nadprodukcja_sebum(datasetToClean, mainOrsecondProblem)
    clean_niedoskonalosci(datasetToClean, mainOrsecondProblem)
    clean_podraznienie(datasetToClean, mainOrsecondProblem)
    clean_przebarwienia(datasetToClean, mainOrsecondProblem)
    clean_rozszerzone_pory(datasetToClean, mainOrsecondProblem)
    clean_suche_skorki(datasetToClean, mainOrsecondProblem)
    clean_szara_cera(datasetToClean, mainOrsecondProblem)
    clean_widoczne_naczynka(datasetToClean, mainOrsecondProblem)    
       
    if mainOrsecondProblem == "Poboczny problem":
        datasetToClean.loc[(datasetToClean["Poboczny problem"] == "brak"), "Poboczny problem"] = "Brak"

def clean_data(datasetToClean):
    '''
    Clean data in dataset
    '''
    datasetToClean = datasetToClean.loc[:, ~datasetToClean.columns.str.contains('^Unnamed')]
    datasetToClean.replace(u'\xa0', u' ', regex=True, inplace=True)
    datasetToClean["Wrażliwa"].fillna(0, inplace=True)
    datasetToClean.dropna(inplace=True)
    clear_skin_type_misspelled_data(datasetToClean)
    clean_main_or_second_problem_misspelled_data(datasetToClean, "Główny problem")
    clean_main_or_second_problem_misspelled_data(datasetToClean, "Poboczny problem")
    datasetToClean.drop_duplicates(inplace=True)
    datasetToClean = datasetToClean[datasetToClean["Wiek"] >= 16]
    return datasetToClean

def remove_punctuation_marks(text):
    '''
    Remove punctuation marks from text
    '''
    text = str(text).replace("'","").replace("[","").replace("]","").replace("\\xa0", " ")
    return text

def clean_product_link(product, PRODUCTS):
    '''
    Clean product link
    '''
    return remove_punctuation_marks(str(PRODUCTS.get(remove_punctuation_marks(product)))).replace("{","").replace("0: ","").replace("}","")
