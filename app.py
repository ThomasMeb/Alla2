import requests
import pandas as pd
pd.options.display.max_columns=200
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
import pickle

data = pd.read_csv("data/data_raw.csv", index_col=0)

# Votre clé API
#api_key = input("Please enter your CryptoCompare API key: ")
api_key = ""



def get_data(existing_data, api_key):
      # Assurez-vous que l'index est de type datetime
      existing_data.index = pd.to_datetime(existing_data.index)

      # Récupère la dernière date dans le DataFrame existant
      last_date = existing_data.index.max()

      # Convertit la dernière date en timestamp UNIX
      last_timestamp = int(time.mktime(last_date.timetuple()))

      # Définit le timestamp pour la fin de la journée actuelle
      end_date = datetime.now().replace(hour=23, minute=59, second=59)
      end_timestamp = int(time.mktime(end_date.timetuple()))

      # Limite de points de données par requête
      limit = 2000

      # Calcule le nombre total de jours et de requêtes nécessaires
      total_days = (end_timestamp - last_timestamp) // (24 * 60 * 60)
      total_requests = -(-total_days // limit)  # Utilise la division entière arrondie vers le haut

      # Fait plusieurs requêtes pour couvrir la période manquante
      for i in range(total_requests):
            # Ajuste le paramètre toTs pour chaque requête
            toTs = end_timestamp - i * limit * 24 * 60 * 60

            # Construit l'URL de l'API
            url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit={limit}&toTs={toTs}&api_key={api_key}'

            # Fait la requête API
            response = requests.get(url)
            data = response.json()

            # Crée un DataFrame avec les données obtenues
            new_data = pd.DataFrame(data['Data']['Data'])
            new_data['time'] = pd.to_datetime(new_data['time'], unit='s')
            new_data.set_index('time', inplace=True)

            # Convertissez les index en format standard pour la comparaison
            existing_data.index = pd.to_datetime(existing_data.index)
            new_data.index = pd.to_datetime(new_data.index)

            # Filtrez les nouvelles données pour ne garder que celles qui ne sont pas dans existing_data
            new_data_filtered = new_data[~new_data.index.isin(existing_data.index)]

            # Fusionnez les données filtrées avec les données existantes
            existing_data = pd.concat([existing_data, new_data_filtered])

      # Triez le DataFrame combiné par index (date)
      existing_data.sort_index(inplace=True)

      return existing_data

# Utilisation de la fonction
data = get_data(data, api_key)
data.drop(['conversionSymbol', 'conversionType'], axis=1, inplace=True)


def calculate_progression(data):
      pd.options.mode.chained_assignment = None  # default='warn'
      data["progression daily"] = 0.0
      data["progression tomorrow"] = 0.0

      # Calcul de la progression du lendemain
      data["progression tomorrow"] = data["close"].values / data["open"].values - 1

      # Calcul de la progression quotidienne
      data.loc[:, "progression daily"] = data['open'].pct_change()

      # Créer une variable cible binaire : 1 si la progression demain est positive, 0 sinon
      data['target'] = np.where(data['progression tomorrow'] > 0, 1, 0)

def calculate_ema(btc_data):
      #Calcul EMA 26

      # Choix du nombre de périodes pour l'EMA. Habituellement, 12, 26 ou 50 sont utilisés.
      n = 26
      alpha = 2 / (n + 1)

      # Calcul de l'EMA
      btc_data['ema_26'] = btc_data['open'].ewm(span=n, adjust=False).mean()

      #Calcul EMA 12

      # Choix du nombre de périodes pour l'EMA. Habituellement, 12, 26 ou 50 sont utilisés.
      n = 12
      alpha = 2 / (n + 1)

      # Calcul de l'EMA
      btc_data['ema_12'] = btc_data['open'].ewm(span=n, adjust=False).mean()

def calculate_macd(btc_data):
      # Calcul du MACD
      btc_data['macd'] = btc_data['ema_12'] - btc_data['ema_26']

def calculate_rsi(btc_data):
      # Calcul du RSI 

      # Calculer la différence de prix par rapport à la journée précédente
      btc_data['delta'] = btc_data['open'].diff()

      # Identifier les gains et les pertes
      btc_data['gain'] = btc_data['delta'].where(btc_data['delta'] > 0, 0)
      btc_data['loss'] = -btc_data['delta'].where(btc_data['delta'] < 0, 0)

      # Calculer la moyenne des gains et des pertes sur 14 jours
      rolling_window = 14
      btc_data['avg_gain'] = btc_data['gain'].rolling(window=rolling_window).mean()
      btc_data['avg_loss'] = btc_data['loss'].rolling(window=rolling_window).mean()

      # Calculer le RS (Relative Strength)
      btc_data['rs'] = btc_data['avg_gain'] / btc_data['avg_loss']

      # Calculer le RSI
      btc_data['rsi'] = 100 - (100 / (1 + btc_data['rs']))

      # Supprimer les colonnes intermédiaires
      btc_data.drop(columns=['delta', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], inplace=True)

def calculate_rel_volume(btc_data):
      # Calcul du volume relatif

      # Période pour calculer la moyenne du volume
      rolling_window = 14

      # Calculer la moyenne du volume sur la période donnée
      btc_data['avg_volume'] = btc_data['volumeto'].rolling(window=rolling_window).mean()

      # Calculer le volume relatif
      btc_data['relative_volume'] = btc_data['volumeto'] / btc_data['avg_volume']

      # Supprimer la colonne de volume moyen intermédiaire si désiré
      btc_data.drop(columns=['avg_volume'], inplace=True)

def calculate_obv(btc_data):
      # Calcul de l'OBV (On-Balance Volume)

      # Calculer la direction du mouvement des prix
      btc_data['price_direction'] = btc_data['open'].diff()

      # Calculer l'OBV
      btc_data['obv'] = btc_data['volumeto'].where(btc_data['price_direction'] > 0, -btc_data['volumeto']).cumsum()

      # Supprimer la colonne intermédiaire de direction des prix
      btc_data.drop(columns=['price_direction'], inplace=True)

def calculate_atr(btc_data):
      # Calcul de l'ATR (Average True Range)

      # Calculer la différence de prix de clôture par rapport à la journée précédente
      btc_data['prev_close'] = btc_data['open'].shift(1)

      # Calculer les trois composantes du True Range
      btc_data['high_minus_low'] = btc_data['high'] - btc_data['low']
      btc_data['high_minus_prev_close'] = abs(btc_data['high'] - btc_data['prev_close'])
      btc_data['low_minus_prev_close'] = abs(btc_data['low'] - btc_data['prev_close'])

      # Déterminer le True Range comme étant le maximum des trois valeurs précédentes
      btc_data['tr'] = btc_data[['high_minus_low', 'high_minus_prev_close', 'low_minus_prev_close']].max(axis=1)

      # Calculer l'ATR comme étant la moyenne mobile du TR sur une période de 14 jours
      rolling_window = 14
      btc_data['atr'] = btc_data['tr'].rolling(window=rolling_window).mean()

      # Supprimer les colonnes intermédiaires
      columns_to_drop = ['prev_close', 'high_minus_low', 'high_minus_prev_close', 'low_minus_prev_close', 'tr']
      btc_data.drop(columns=columns_to_drop, inplace=True)

def calculate_bollinger(btc_data):
      # Calcul des bandes de Bollinger

      rolling_window = 20

      # Moyenne mobile simple
      btc_data['sma'] = btc_data['open'].rolling(window=rolling_window).mean()

      # Écart-type des prix sur la période
      btc_data['price_std'] = btc_data['open'].rolling(window=rolling_window).std()

      # Calcul des bandes de Bollinger
      btc_data['bollinger_upper'] = btc_data['sma'] + (btc_data['price_std'] * 2)
      btc_data['bollinger_lower'] = btc_data['sma'] - (btc_data['price_std'] * 2)

      # Suppression des colonnes intermédiaires
      btc_data.drop(columns=['sma', 'price_std'], inplace=True)

def calculate_stoch_osc(btc_data):
      # Oscillateur Stochastique

      rolling_window = 14

      # Trouver le prix le plus bas et le plus élevé sur la période
      btc_data['rolling_low'] = btc_data['low'].rolling(window=rolling_window).min()
      btc_data['rolling_high'] = btc_data['high'].rolling(window=rolling_window).max()

      # Calcul du Stochastic Oscillator
      btc_data['k'] = 100 * ((btc_data['open'] - btc_data['rolling_low']) / (btc_data['rolling_high'] - btc_data['rolling_low']))

      # Suppression des colonnes intermédiaires
      btc_data.drop(columns=['rolling_low', 'rolling_high'], inplace=True)

def calculate_momentum(btc_data):
      # Momentum

      n_days = 10

      # Calcul du Momentum
      btc_data['momentum'] = btc_data['open'] - btc_data['open'].shift(n_days)

def calculate_features(data):
      calculate_progression(data)
      calculate_ema(data)
      calculate_macd(data)
      calculate_rsi(data)
      calculate_rel_volume(data)
      calculate_obv(data)
      calculate_atr(data)
      calculate_bollinger(data)
      calculate_stoch_osc(data)
      calculate_momentum(data)

calculate_features(data)

start_date = pd.to_datetime("2011-01-01")
data = data[data.index >= start_date]

"""print(data)
print("valeurs nulles : " , data.isna().sum())
duplicates = data.index.duplicated(keep=False)
print("doublons : ", data[duplicates])
print(data.loc[data['volumefrom']==0])"""


xgboost = pickle.load(open("models/xgboost_model.pkl", 'rb'))

# Sélectionner les caractéristiques et exclure la dernière ligne
features = data.drop(columns=['progression tomorrow', 'target', 'close', 'high', 'low', 'volumefrom']).iloc[:-1, :]
target = data['target'].iloc[:-1]
window_size = 1500

def predict_with_model(model, scaler, features, window_size):
      # S'assurer que les features sont dans le bon format
      features = pd.DataFrame(features)

      # Vérifier si le DataFrame a suffisamment de lignes
      if len(features) < window_size:
            raise ValueError("Le DataFrame features n'a pas assez de lignes par rapport à window_size")

      # Préparer les données de test pour le dernier segment
      X_test = features.iloc[-window_size:, :]

      # Normaliser les données
      scaler = StandardScaler()
      X_test = scaler.fit_transform(X_test)

      # Obtenir les probabilités prédites pour la classe positive 
      prediction_prob = model.predict_proba(X_test)[:, 1]

      # Retourner la probabilité prédite pour le dernier jour
      return prediction_prob[-1]

#predicted_probabilities = predict_with_model(xgboost, features, window_size)
#print (predicted_probabilities)


"""wallet_baseline = [1000, 0]
wallet_test = [1000, 0]
total = sum(wallet_test)

wallet_test[0] = (total * predicted_probabilities)
wallet_test[1] = total - wallet_test[0]

wallet_test[0] *= (data['progression tomorrow'].iloc[-1]+1)"""




def get_past_dates(n):
    # Liste pour stocker les dates
    past_dates = []

    # Obtenir la date d'hier
    yesterday = datetime.now() - timedelta(days=15)

    # Boucle pour obtenir les dates de "hier à il y a n jours"
    for i in range(n-1, 0, -1):
        # Calcul de la date
        date = yesterday - timedelta(days=i)

        # Formatage de la date en 'année-mois-jour'
        formatted_date = date.strftime('%Y-%m-%d')

        # Ajouter la date formatée à la liste
        past_dates.append(formatted_date)

    # Ajouter hier à la liste
    past_dates.append(yesterday.strftime('%Y-%m-%d'))

    return past_dates

n_days = 31
dates = get_past_dates(n_days)
print(dates)

xgb_models = {}
xgb_scalers = {}
predictions = {}
wallet_baseline = [1000, 0]
wallet_test = [1000, 0]
ratio = 0

for date in dates:
      model_filename = f"models/xgboost_models/xgboost_{date}.pkl"
      with open(model_filename, 'rb') as file:
            xgb_models[date] = pickle.load(file)
      scaler_filename = f"scalers/scaler_{date}.pkl"
      with open(scaler_filename, 'rb') as file:
            xgb_scalers[date] = pickle.load(file)
      #print(model_filename)

      # Filtrer 'data' pour ne garder que les données jusqu'à la date concernée
      filtered_data = data[data.index <= date]
      #print(filtered_data[-1:]["progression tomorrow"])  

      # Préparer les features pour la date filtrée
      features = filtered_data.drop(columns=['progression tomorrow', 'target', 'close', 'high', 'low', 'volumefrom']).iloc[:-1, :]

      # Faire la prédiction avec le modèle chargé     
      prediction = predict_with_model(xgb_models[date], xgb_scalers[date], features, window_size)
      #print(prediction)

      progression = filtered_data["progression tomorrow"].iloc[-1]+1

      total = sum(wallet_test)
      wallet_test[0] = (total * prediction)
      wallet_test[1] = total - wallet_test[0]
      wallet_test[0] *= (progression)
      wallet_baseline[0] *= (progression)
      total = sum(wallet_test)

      if (prediction < 0.5 and progression<1) or (prediction>0.5 and progression>1):
            ratio+=1
      else:
            ratio-=1

      print(f'date : {date}, predi : {prediction.round(2)}, prog : {100*(progression-1).round(5)}%, base : {wallet_baseline}, test : {wallet_test}, total : {total.round(2)}, ratio : {ratio}')
      # Stocker la prédiction
      predictions[date] = prediction

      
#print(predictions)