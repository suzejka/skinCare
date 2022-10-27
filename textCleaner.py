def clear_skin_type_misspelled_data(datasetToClean):
    datasetToClean.loc[(datasetToClean["Typ cery"] == "tłusta") | 
                        (datasetToClean["Typ cery"] == "tlusta") | 
                        (datasetToClean["Typ cery"] == "Tlusta"), "Typ cery"] = "Tłusta"
    datasetToClean.loc[(datasetToClean["Typ cery"] == "mieszana"), "Typ cery"] = "Mieszana"
    datasetToClean.loc[(datasetToClean["Typ cery"] == "sucha"), "Typ cery"] = "Sucha"
    datasetToClean.loc[(datasetToClean["Typ cery"] == "normalna"), "Typ cery"] = "Normalna"

def clean_nadprodukcja_sebum(datasetToClean, mainOrsecondProblem):
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "nadprodukcja sebum") | 
                        (datasetToClean[mainOrsecondProblem] == "nadprodukcja Sebum"), mainOrsecondProblem] = "Nadprodukcja sebum"

def clean_niedoskonalosci(datasetToClean, mainOrsecondProblem):
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "niedoskonałości") | 
                        (datasetToClean[mainOrsecondProblem] == "niedoskonalosci") |
                        (datasetToClean[mainOrsecondProblem] == "niedoskonałosci") |
                        (datasetToClean[mainOrsecondProblem] == "niedoskonalości") |
                        (datasetToClean[mainOrsecondProblem] == "Niedoskonalości") | 
                        (datasetToClean[mainOrsecondProblem] == "Niedoskonalosci") |
                        (datasetToClean[mainOrsecondProblem] == "Niedoskonałosci"), mainOrsecondProblem] = "Niedoskonałości"
            
def clean_podraznienie(datasetToClean, mainOrsecondProblem):
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "podrażnienie") | 
                        (datasetToClean[mainOrsecondProblem] == "podraznienie") |
                        (datasetToClean[mainOrsecondProblem] == "Podraznienie") , mainOrsecondProblem] = "Podrażnienie"
                    
def clean_przebarwienia(datasetToClean, mainOrsecondProblem):
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "przebarwienia"), mainOrsecondProblem] = "Przebarwienia"

def clean_rozszerzone_pory(datasetToClean, mainOrsecondProblem):
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "rozszerzone pory") | 
                        (datasetToClean[mainOrsecondProblem] == "Rozszerzone Pory") |
                        (datasetToClean[mainOrsecondProblem] == "rozszerzone Pory") , mainOrsecondProblem] = "Rozszerzone pory"

def clean_suche_skorki(datasetToClean, mainOrsecondProblem):
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "suche skórki") | 
                        (datasetToClean[mainOrsecondProblem] == "suche Skórki") |
                        (datasetToClean[mainOrsecondProblem] == "Suche skorki") |
                        (datasetToClean[mainOrsecondProblem] == "suche skorki") |
                        (datasetToClean[mainOrsecondProblem] == "suche Skorki"), mainOrsecondProblem] = "Suche skórki"

def clean_szara_cera(datasetToClean, mainOrsecondProblem):
    datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "szara cera") | 
                        (datasetToClean[mainOrsecondProblem] == "szara Cera") |
                        (datasetToClean[mainOrsecondProblem] == "Szara Cera"), mainOrsecondProblem] = "Szara cera"

def clean_widoczne_naczynka(datasetToClean, mainOrsecondProblem):
     datasetToClean.loc[(datasetToClean[mainOrsecondProblem] == "widoczne naczynka") | 
                        (datasetToClean[mainOrsecondProblem] == "Widoczne Naczynka") |
                        (datasetToClean[mainOrsecondProblem] == "widoczne Naczynka"), mainOrsecondProblem] = "Widoczne naczynka"

def clear_main_or_second_problem_misspelled_data(datasetToClean, mainOrsecondProblem):
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

def clear_data(datasetToClean):
    datasetToClean.dropna(inplace=True)
    clear_skin_type_misspelled_data(datasetToClean)
    clear_main_or_second_problem_misspelled_data(datasetToClean, "Główny problem")
    clear_main_or_second_problem_misspelled_data(datasetToClean, "Poboczny problem")
    datasetToClean.drop_duplicates(inplace=True)
    datasetToClean = datasetToClean[datasetToClean["Wiek"] >= 16]

