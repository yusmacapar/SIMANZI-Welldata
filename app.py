import os
import joblib 
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify 

# --- 1. SETUP & MUAT MODEL ---

app = Flask(__name__)

K_OPTIMAL = 7
MODEL_DIR = 'model_files'

# Fungsi untuk mencari nama kolom yang mengandung kata kunci (case-insensitive)
def find_column_key(df, keywords):
    """Mencari nama kolom dalam DataFrame yang mengandung semua kata kunci."""
    keywords_lower = [k.lower() for k in keywords]
    for col in df.columns:
        # Menangani kolom dengan nama 'Protein.1', 'Carbohydrates.1', dsb.
        col_lower = col.lower().replace('.1', '') 
        
        # Cek apakah semua kata kunci ada di nama kolom
        if all(kw in col_lower for kw in keywords_lower):
            return col
    return None

try:
    PREPROCESSOR = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    # Y_DATA_TRAIN harus mengandung kolom Menu (string) dan Nutrisi (numerik)
    Y_DATA_TRAIN = joblib.load(os.path.join(MODEL_DIR, 'y_data.pkl')) 
    KNN_MODEL = joblib.load(os.path.join(MODEL_DIR, 'knn_model.pkl'))
    print("Artefak model berhasil dimuat. K-Optimal:", K_OPTIMAL)
    print("Kolom Y_DATA_TRAIN:", Y_DATA_TRAIN.columns.tolist()) 
except FileNotFoundError:
    print(f"Error: File model tidak ditemukan di {MODEL_DIR}. Pastikan folder ada dan isinya lengkap.")
    # Exit di Flask development environment
    # raise FileNotFoundError(f"File model tidak ditemukan di {MODEL_DIR}")
    exit()

# Kolom input yang dibutuhkan, harus sesuai urutan saat training!
EXPECTED_COLUMNS = [
    'Ages', 'Height', 'Weight', 'Daily Calorie Target', 'Protein', 
    'Carbohydrates', 'Fiber', 'Fat', 'Gender', 'Activity Level', 
    'Dietary Preference', 'Disease'
]

# Definisikan kolom nutrisi agar mudah diakses
NUTRIENT_COLS = [
    'Breakfast Calories', 'Breakfast Protein', 'Breakfast Carbohydrates', 'Breakfast Fats',
    'Lunch Calories', 'Lunch Protein', 'Lunch Carbohydrates', 'Lunch Fats',
    'Dinner Calories', 'Dinner Protein.1', 'Dinner Carbohydrates.1', 'Dinner Fats',
    'Snacks Calories', 'Snacks Protein', 'Snacks Carbohydrates', 'Snacks Fats'
]

# --- 2. ENDPOINT UTAMA & HELPER BARU ---

def calculate_best_scale_and_deviation(row_dict, target_kalori, targets):
    """
    Menghitung faktor skala untuk mencapai total kalori target 
    dan deviasi makronutrien absolut dari menu yang diskala.
    
    Digunakan untuk memilih menu terbaik dari K-Nearest Neighbors.
    """
    
    # 1. Hitung total kalori UN-SCALED dari menu 4 porsi
    unscaled_kalori_sum = 0.0
    # Menggunakan Y_DATA_TRAIN untuk menemukan nama kolom nutrisi
    for meal_type in ["Breakfast", "Lunch", "Dinner", "Snacks"]:
        kalori_col = find_column_key(Y_DATA_TRAIN, [meal_type, 'Calories'])
        unscaled_kalori_sum += float(row_dict.get(kalori_col, 0.0))

    if unscaled_kalori_sum == 0 or target_kalori == 0:
        return float('inf'), 0.0 # Deviasi tak terhingga jika tidak ada kalori

    # 2. Tentukan faktor skala: Target Kalori / Total Kalori Menu Asli
    scale_factor = target_kalori / unscaled_kalori_sum
    
    MAX_SCALE_FACTOR = 2.0 
    if scale_factor > MAX_SCALE_FACTOR:
        scale_factor = MAX_SCALE_FACTOR
    # Jika menu asli sudah sedikit melebihi target (misal > 105%), batasi skala kembali 
    elif unscaled_kalori_sum > target_kalori * 1.05:
         scale_factor = 1.05

    # 3. Hitung Makro yang Di-skala dan Deviasi
    total_protein_scaled = 0.0
    total_carbo_scaled = 0.0
    total_fat_scaled = 0.0
    
    for meal_type in ["Breakfast", "Lunch", "Dinner", "Snacks"]:
        protein_col = find_column_key(Y_DATA_TRAIN, [meal_type, 'Protein'])
        carbo_col = find_column_key(Y_DATA_TRAIN, [meal_type, 'Carbohydrates']) 
        fat_col = find_column_key(Y_DATA_TRAIN, [meal_type, 'Fats']) 

        total_protein_scaled += float(row_dict.get(protein_col, 0.0)) * scale_factor
        total_carbo_scaled += float(row_dict.get(carbo_col, 0.0)) * scale_factor
        total_fat_scaled += float(row_dict.get(fat_col, 0.0)) * scale_factor

    # 4. Hitung Total Deviasi Absolut
    dev_protein = abs(total_protein_scaled - targets['Protein'])
    dev_carbo = abs(total_carbo_scaled - targets['Carbohydrates'])
    dev_fat = abs(total_fat_scaled - targets['Fat'])
    
    # Berikan bobot lebih pada Deviasi Lemak agar sistem memilih menu yang lebih rendah lemak
    # Mengalikan deviasi lemak dengan 1.5 sebagai bobot tambahan.
    total_deviation = dev_protein + dev_carbo + (dev_fat * 1.5) 
    
    return total_deviation, scale_factor

@app.route("/")
def home():
    """Merender halaman form input."""
    return render_template('index.html') 

@app.route("/rekomendasi", methods=["POST"])
def rekomendasi():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Permintaan harus berupa JSON yang valid."}), 400

        # Ambil data input mentah
        form_data = {
            'Ages': float(data.get('Ages', 30)),
            'Height': float(data.get('Height', 170)),
            'Weight': float(data.get('Weight', 65)),
            'Gender': data.get('Gender', 'Male'),
            'Activity Level': data.get('Activity Level', 'Moderately Active'),
            'Daily Calorie Target': float(data.get('Daily Calorie Target', 2000)),
            'Protein': float(data.get('Protein', 100)),
            'Carbohydrates': float(data.get('Carbohydrates', 250)),
            'Fiber': float(data.get('Fiber', 30)),
            'Fat': float(data.get('Fat', 60)),
            'Dietary Preference': data.get('Dietary Preference', 'Omnivore'),
            'Disease': data.get('Disease', 'Tidak Ada')
        }
        
        input_data = {col: form_data.get(col) for col in EXPECTED_COLUMNS}
        input_df = pd.DataFrame([input_data])

        # 1. Pra-pemrosesan Input Baru
        input_processed = PREPROCESSOR.transform(input_df)

        # 2. Cari K-Tetangga Terdekat (K=7)
        distances, indices = KNN_MODEL.kneighbors(input_processed, n_neighbors=K_OPTIMAL)
        
        # 3. EKSTRAKSI TOTAL TARGET HARIAN DARI DATA INPUT USER
        target_kalori_float = form_data['Daily Calorie Target'] # Ambil float untuk kalkulasi
        target_macros_float = {
            'Protein': form_data['Protein'],
            'Carbohydrates': form_data['Carbohydrates'],
            'Fat': form_data['Fat']
        }
        
        # 4. PILIH TETANGGA TERBAIK BERDASARKAN KESESUAIAN MAKRO SETELAH SCALING
        # Iterasi melalui K-tetangga dan pilih yang memiliki deviasi makro terkecil setelah diskala kalori
        best_deviation = float('inf')
        best_row_index = -1
        final_scale_factor = 1.0

        for index in indices[0]:
            row_dict = Y_DATA_TRAIN.iloc[index].to_dict()
            deviation, current_scale_factor = calculate_best_scale_and_deviation(
                row_dict, target_kalori_float, target_macros_float
            )

            # Pilih menu dengan deviasi makro terkecil. 
            # (Tidak perlu membandingkan jarak KNN di sini, karena prioritas utama adalah kesesuaian makro)
            if deviation < best_deviation:
                best_deviation = deviation
                best_row_index = index
                final_scale_factor = current_scale_factor
            
            # Tetapkan tetangga pertama sebagai fallback jika tidak ada yang ditemukan
            if best_row_index == -1: 
                best_row_index = indices[0][0]
                # Hitung ulang scale factor untuk fallback
                fallback_row_dict = Y_DATA_TRAIN.iloc[best_row_index].to_dict()
                _, final_scale_factor = calculate_best_scale_and_deviation(
                    fallback_row_dict, target_kalori_float, target_macros_float
                )


        rekomendasi_row = Y_DATA_TRAIN.iloc[best_row_index].to_dict()
        scale_factor = final_scale_factor

        # 5. Format Total Target Harian (hanya formatting string)
        targets = {
            "kalori_target": f"{target_kalori_float:.1f}",
            "protein_target": f"{form_data['Protein']:.1f}",
            "karbo_target": f"{form_data['Carbohydrates']:.1f}", 
            "lemak_target": f"{form_data['Fat']:.1f}",
        }
        total_targets = targets

        # 6. Format hasil menu dan hitung total menu yang DI-SKALA
        menu_result = {} 
        total_menu_nutrisi = {
            "Total Kalori": 0.0,
            "Total Protein": 0.0,
            "Total Karbohidrat": 0.0,
            "Total Lemak": 0.0,
        }
        
        menu_keys = {
            "Sarapan": {
                'nama': ['Breakfast', 'Suggestion'], 
                'kalori': ['Breakfast', 'Calories'], 
                'protein': ['Breakfast', 'Protein'],
                'carbohydrates': ['Breakfast', 'Carbohydrates'], 
                'fat': ['Breakfast', 'Fats'] 
            },
            "Makan Siang": {
                'nama': ['Lunch', 'Suggestion'], 
                'kalori': ['Lunch', 'Calories'], 
                'protein': ['Lunch', 'Protein'],
                'carbohydrates': ['Lunch', 'Carbohydrates'], 
                'fat': ['Lunch', 'Fats'] 
            },
            "Makan Malam": {
                'nama': ['Dinner', 'Suggestion'], 
                'kalori': ['Dinner', 'Calories'], 
                'protein': ['Dinner', 'Protein'], 
                'carbohydrates': ['Dinner', 'Carbohydrates'], 
                'fat': ['Dinner', 'Fats'] 
            },
            "Camilan": {
                'nama': ['Snack', 'Suggestion'], 
                'kalori': ['Snacks', 'Calories'], 
                'protein': ['Snacks', 'Protein'],
                'carbohydrates': ['Snacks', 'Carbohydrates'], 
                'fat': ['Snacks', 'Fats'] 
            }
        }

        for meal, keys in menu_keys.items():
            nama_col = find_column_key(Y_DATA_TRAIN, keys['nama'])
            kalori_col = find_column_key(Y_DATA_TRAIN, keys['kalori'])
            protein_col = find_column_key(Y_DATA_TRAIN, keys['protein'])
            carbohydrates_col = find_column_key(Y_DATA_TRAIN, keys['carbohydrates']) 
            fat_col = find_column_key(Y_DATA_TRAIN, keys['fat']) 
            
            # Ambil nilai UN-SCALED
            kalori_unscaled = rekomendasi_row.get(kalori_col, 0.0)
            protein_unscaled = rekomendasi_row.get(protein_col, 0.0)
            carbo_unscaled = rekomendasi_row.get(carbohydrates_col, 0.0)
            fat_unscaled = rekomendasi_row.get(fat_col, 0.0)

            # Terapkan faktor skala
            kalori_scaled = float(kalori_unscaled) * scale_factor
            protein_scaled = float(protein_unscaled) * scale_factor
            carbo_scaled = float(carbo_unscaled) * scale_factor
            fat_scaled = float(fat_unscaled) * scale_factor

            # Hitung total nutrisi menu (Menggunakan nilai yang di-skala)
            total_menu_nutrisi["Total Kalori"] += kalori_scaled
            total_menu_nutrisi["Total Protein"] += protein_scaled
            total_menu_nutrisi["Total Karbohidrat"] += carbo_scaled
            total_menu_nutrisi["Total Lemak"] += fat_scaled
            
            menu_result[meal] = {
                "nama": rekomendasi_row.get(nama_col, "Menu not found"),
                "kalori": f"{kalori_scaled:.1f}",
                "protein": f"{protein_scaled:.1f}",
                "carbohydrates": f"{carbo_scaled:.1f}", 
                "fat": f"{fat_scaled:.1f}", 
            }
        
        # 7. Siapkan hasil akhir
        distance = distances[0][0] # Gunakan jarak ke tetangga terdekat (index 0) untuk status kemiripan
        if distance < 0.01:
            profil_mirip_status = "Sangat Mirip (Data di Dataset hampir identik dengan profil Anda)"
        elif distance < 0.5:
            profil_mirip_status = f"Mirip (Jarak: {distance:.2f})"
        else:
            profil_mirip_status = f"Cukup Mirip (Jarak: {distance:.2f})"

        # 8. Hitung Deviasi Makro (untuk informasi tambahan)
        target_protein = form_data['Protein']
        target_carbo = form_data['Carbohydrates']
        target_fat = form_data['Fat']
        
        deviasi = {
            "Protein": total_menu_nutrisi["Total Protein"] - target_protein,
            "Karbohidrat": total_menu_nutrisi["Total Karbohidrat"] - target_carbo,
            "Lemak": total_menu_nutrisi["Total Lemak"] - target_fat,
        }

        result = {
            "profil_mirip_jarak": profil_mirip_status,
            "target_harian": total_targets, 
            "rekomendasi": menu_result, 
            "total_menu_nutrisi": {
                "Total Kalori": f"{total_menu_nutrisi['Total Kalori']:.1f}",
                "Total Protein": f"{total_menu_nutrisi['Total Protein']:.1f}",
                "Total Karbohidrat": f"{total_menu_nutrisi['Total Karbohidrat']:.1f}",
                "Total Lemak": f"{total_menu_nutrisi['Total Lemak']:.1f}",
            },
            "scale_factor": f"{scale_factor:.2f}", # Tambahkan faktor skala
            "macro_deviation": deviasi, # Tambahkan deviasi makro
            "user_data": {
                "berat": form_data['Weight'],
                "tinggi": form_data['Height']
            }
        }
        
        return jsonify(result)

    except Exception as e:
        error_message = f"Terjadi kesalahan saat memproses data: {str(e)}"
        print("ERROR PENGAMBILAN DATA ATAU MODEL:", error_message) 
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    # Gunakan host dan port yang sesuai untuk development
    app.run(debug=True, host='0.0.0.0', port=5000)
