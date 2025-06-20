import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Fixer la seed pour la reproductibilité
np.random.seed(42)
random.seed(42)

def generate_carbon_footprint_data(n_samples=1000):
    """
    Génère des données fictives réalistes pour un calculateur d'empreinte carbone
    avec préfixes par catégorie
    """
    
    codes_postaux_bruxelles = [
        "1000",  # Bruxelles (centre-ville)
        "1020",  # Bruxelles (Laeken)
        "1030",  # Schaerbeek
        "1040",  # Etterbeek
        "1050",  # Ixelles
        "1060",  # Saint-Gilles
        "1070",  # Anderlecht
        "1080",  # Molenbeek-Saint-Jean
        "1081",  # Koekelberg
        "1082",  # Berchem-Sainte-Agathe
        "1083",  # Ganshoren
        "1090",  # Jette
        "1120",  # Neder-Over-Heembeek (partie de Bruxelles)
        "1130",  # Haren (partie de Bruxelles)
        "1140",  # Evere
        "1150",  # Woluwe-Saint-Pierre
        "1160",  # Auderghem
        "1170",  # Watermael-Boitsfort
        "1180",  # Uccle
        "1190",  # Forest
        "1200",  # Woluwe-Saint-Lambert
        "1210"   # Saint-Josse-ten-Noode
    ]

    # Listes de valeurs réalistes
    # Pondération pour rendre certains codes plus fréquents
    weights = [10, 8, 7, 7, 8, 6, 6, 5, 2, 2, 2, 4, 2, 2, 3, 3, 3, 3, 5, 4, 4, 2]
    weights = [w / sum(weights) for w in weights]  # Normalisation
    codes_postaux = random.choices(codes_postaux_bruxelles, weights=weights, k=100)

    types_logement = ["Appartement", "Maison", "Studio", "Colocation"]
    types_chauffage = ["Gaz", "Électrique", "Fioul", "Pompe à chaleur", "Bois"]
    isolation_options = ["Très bien isolé", "Bien isolé", "Moyennement isolé", "Mal isolé"]
    qualites_video = ["SD", "HD", "4K"]
    
    data = []
    
    for i in range(n_samples):
        # Date de participation (6 derniers mois)
        date_participation = datetime.now() - timedelta(days=random.randint(0, 180))
        
        # Informations de base
        code_postal = random.choice(codes_postaux)
        
        # === SE LOGER ===
        type_logement = random.choice(types_logement)
        
        # Superficie selon le type de logement
        if type_logement == "Studio":
            superficie = random.randint(20, 40)
        elif type_logement == "Appartement":
            superficie = random.randint(40, 120)
        elif type_logement == "Maison":
            superficie = random.randint(80, 200)
        else:  # Colocation
            superficie = random.randint(60, 150)
        
        nb_habitants = random.choices([1, 2, 3, 4, 5, 6], weights=[20, 35, 20, 15, 7, 3])[0]
        type_chauffage = random.choice(types_chauffage)
        isolation = random.choice(isolation_options)
        temperature_chauffage = random.randint(18, 23)
        
        # === SE DEPLACER ===
        km_voiture = random.randint(0, 25000)
        km_voiture_electrique = random.randint(0, 15000) if random.random() < 0.3 else 0
        km_train = random.randint(0, 8000)
        km_bus = random.randint(0, 5000)
        km_metro = random.randint(0, 3000)
        km_tram = random.randint(0, 2000)
        km_velo = random.randint(0, 2000)
        km_velo_electrique = random.randint(0, 3000) if random.random() < 0.25 else 0
        km_trottinette = random.randint(0, 500)
        km_moto = random.randint(0, 8000) if random.random() < 0.15 else 0
        nb_vols_avion = random.choices([0, 1, 2, 3, 4, 5, 6], weights=[30, 25, 20, 15, 7, 2, 1])[0]
        
        # === SE NOURRIR ===
        repas_viande_semaine = random.randint(0, 14)  # Sur 14 repas principaux
        eau_bouteille = random.choice(["Oui", "Non"])
        petits_electromenager_cuisine = random.randint(2, 8)
        gros_electromenager_cuisine = random.randint(3, 6)
        freq_remplacement_cuisine = random.choice(["Moins de 5 ans", "5-10 ans", "10-15 ans", "Plus de 15 ans"])
        
        # === S'HABILLER ===
        vetements_neufs_an = random.randint(5, 50)
        
        # === SE DIVERTIR ===
        petits_electromenager_menage = random.randint(2, 8)
        gros_electromenager_menage = random.randint(2, 6)
        freq_remplacement_menage = random.choice(["Moins de 5 ans", "5-10 ans", "10-15 ans", "Plus de 15 ans"])
        appareils_numeriques = random.randint(2, 10)
        heures_video_jour = random.randint(0, 8)
        qualite_video = random.choice(qualites_video)
        
        # === CALCUL DES IMPACTS (en tonnes CO2) ===
        # Facteurs d'émission simplifiés
        
        # Impact logement
        impact_logement = (
            superficie * 0.02 +  # Impact surface
            (25 - temperature_chauffage) * 0.1 +  # Impact température
            {"Gaz": 1.5, "Électrique": 0.8, "Fioul": 2.0, "Pompe à chaleur": 0.6, "Bois": 0.4}[type_chauffage] +
            {"Très bien isolé": -0.3, "Bien isolé": -0.1, "Moyennement isolé": 0.1, "Mal isolé": 0.3}[isolation]
        ) / nb_habitants
        
        # Impact transport
        impact_transport = (
            km_voiture * 0.0002 +
            km_voiture_electrique * 0.00005 +
            km_train * 0.000014 +
            km_bus * 0.00008 +
            km_metro * 0.000005 +
            km_tram * 0.000005 +
            km_velo * 0 +
            km_velo_electrique * 0.000001 +
            km_trottinette * 0.0001 +
            km_moto * 0.0001 +
            nb_vols_avion * 0.5
        )
        
        # Impact alimentation
        impact_alimentation = (
            repas_viande_semaine * 0.05 +
            (14 - repas_viande_semaine) * 0.01 +
            (1 if eau_bouteille == "Oui" else 0) * 0.1 +
            petits_electromenager_cuisine * 0.01 +
            gros_electromenager_cuisine * 0.05
        )
        
        # Impact vêtements
        impact_vetements = vetements_neufs_an * 0.02
        
        # Impact divertissement
        impact_divertissement = (
            petits_electromenager_menage * 0.0025 +
            gros_electromenager_menage * 0.0125 +
            appareils_numeriques * 0.0125 +
            heures_video_jour * 0.0025 * 365 +
            {"SD": 0, "HD": 0.025, "4K": 0.075}[qualite_video]
        )
        
        # Impact total
        impact_total = impact_logement + impact_transport + impact_alimentation + impact_vetements + impact_divertissement
        
        # Ajout de variabilité réaliste
        impact_total *= random.uniform(0.8, 1.2)
        
        # Empreinte carbone moyenne (simulation)
        empreinte_moyenne = round(random.uniform(8, 12), 2)
        
        data.append({
            # Informations générales (sans préfixe)
            'date_participation': date_participation.strftime('%Y-%m-%d'),
            'code_postal': code_postal,
            'empreinte_moyenne': empreinte_moyenne,
            
            # SE LOGER (préfixe: logement_)
            'logement_type': type_logement,
            'logement_superficie_m2': superficie,
            'logement_nb_habitants': nb_habitants,
            'logement_type_chauffage': type_chauffage,
            'logement_isolation': isolation,
            'logement_temperature_chauffage': temperature_chauffage,
            
            # SE DEPLACER (préfixe: transport_)
            'transport_km_voiture': km_voiture,
            'transport_km_voiture_electrique': km_voiture_electrique,
            'transport_km_train': km_train,
            'transport_km_bus': km_bus,
            'transport_km_metro': km_metro,
            'transport_km_tram': km_tram,
            'transport_km_velo': km_velo,
            'transport_km_velo_electrique': km_velo_electrique,
            'transport_km_trottinette': km_trottinette,
            'transport_km_moto': km_moto,
            'transport_nb_vols_avion': nb_vols_avion,
            
            # SE NOURRIR (préfixe: alimentation_)
            'alimentation_repas_viande_semaine': repas_viande_semaine,
            'alimentation_eau_bouteille': eau_bouteille,
            'alimentation_petits_electromenager': petits_electromenager_cuisine,
            'alimentation_gros_electromenager': gros_electromenager_cuisine,
            'alimentation_freq_remplacement': freq_remplacement_cuisine,
            
            # S'HABILLER (préfixe: vetements_)
            'vetements_neufs_an': vetements_neufs_an,
            
            # SE DIVERTIR (préfixe: divertissement_)
            'divertissement_petits_electromenager': petits_electromenager_menage,
            'divertissement_gros_electromenager': gros_electromenager_menage,
            'divertissement_freq_remplacement': freq_remplacement_menage,
            'divertissement_appareils_numeriques': appareils_numeriques,
            'divertissement_heures_video_jour': heures_video_jour,
            'divertissement_qualite_video': qualite_video,
            
            # IMPACTS (préfixe: impact_)
            'impact_logement': round(max(0, impact_logement), 2),
            'impact_transport': round(max(0, impact_transport), 2),
            'impact_alimentation': round(max(0, impact_alimentation), 2),
            'impact_vetements': round(max(0, impact_vetements), 2),
            'impact_divertissement': round(max(0, impact_divertissement), 2),
            'impact_total': round(max(0, impact_total), 2)
        })
    
    return pd.DataFrame(data)

# Génération des données
df = generate_carbon_footprint_data(2471)

# Sauvegarde
df.to_csv('empreinte_carbone_data.csv', index=False)

# Fonction pour analyser les colonnes par préfixe
def analyze_by_prefix(df):
    """Analyse les colonnes par préfixe"""
    prefixes = {
        'Générales': [col for col in df.columns if not any(col.startswith(p) for p in ['logement_', 'transport_', 'alimentation_', 'vetements_', 'divertissement_', 'impact_'])],
        'Logement': [col for col in df.columns if col.startswith('logement_')],
        'Transport': [col for col in df.columns if col.startswith('transport_')],
        'Alimentation': [col for col in df.columns if col.startswith('alimentation_')],
        'Vêtements': [col for col in df.columns if col.startswith('vetements_')],
        'Divertissement': [col for col in df.columns if col.startswith('divertissement_')],
        'Impacts': [col for col in df.columns if col.startswith('impact_')]
    }
    
    return prefixes

# Affichage des résultats
print("Données générées avec succès !")
print(f"\nNombre d'échantillons : {len(df)}")
print(f"\nNombre total de colonnes : {len(df.columns)}")

# Analyse par préfixe
prefixes = analyze_by_prefix(df)
print("\n" + "="*50)
print("COLONNES PAR CATÉGORIE")
print("="*50)

for category, columns in prefixes.items():
    print(f"\n{category.upper()} ({len(columns)} colonnes):")
    for col in columns:
        print(f"  - {col}")

print(f"\n" + "="*50)
print("APERÇU DES DONNÉES")
print("="*50)
print(df.head(3))

print(f"\n" + "="*50)
print("STATISTIQUES DES IMPACTS")
print("="*50)
impact_columns = [col for col in df.columns if col.startswith('impact_')]
print(df[impact_columns].describe())

print(f"\n" + "="*50)
print("EXEMPLES DE RÉPARTITIONS")
print("="*50)
print(f"\nRépartition logement_type :")
print(df['logement_type'].value_counts())

print(f"\nRépartition logement_type_chauffage :")
print(df['logement_type_chauffage'].value_counts())

print(f"\nRépartition alimentation_eau_bouteille :")
print(df['alimentation_eau_bouteille'].value_counts())

print(f"\nRépartition divertissement_qualite_video :")
print(df['divertissement_qualite_video'].value_counts())

# Fonction utilitaire pour filtrer les colonnes par préfixe
def get_columns_by_prefix(df, prefix):
    """Retourne les colonnes ayant le préfixe spécifié"""
    return [col for col in df.columns if col.startswith(prefix)]

# Exemples d'utilisation
print(f"\n" + "="*50)
print("FONCTIONS UTILITAIRES")
print("="*50)
print(f"Colonnes logement: {get_columns_by_prefix(df, 'logement_')}")
print(f"Colonnes transport: {len(get_columns_by_prefix(df, 'transport_'))} colonnes")
print(f"Colonnes alimentation: {len(get_columns_by_prefix(df, 'alimentation_'))} colonnes")
