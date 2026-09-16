import os
import sys
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, redirect
from flaskwebgui import FlaskUI 

def get_resource_path(relative_path):
    try: base_path = sys._MEIPASS
    except: base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

app = Flask(__name__, 
            template_folder=get_resource_path('templates'),
            static_folder=get_resource_path('static'))

MODEL_PATH = get_resource_path('decision_tree_model.pkl')
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# GLOBAL CONFIGURATION
config = {
    "density_relation": 0.02, 
    "uncoated_factor": 0.75,   # New: Uncoated paper only gets 75% density per % of ink
    "zero_setting": 0.0,
    "target_de": 2.5
}

@app.route('/')
def index():
    '''
    Returns the main page (index.html located in /templates) with the current configuration values passed in for display and use in the frontend. 
    The config dictionary contains:
    - density_relation: The base density change per percentage of ink key change.   
    - uncoated_factor: A multiplier applied to the density_relation for uncoated paper types, reflecting their lower ink absorption.
    - zero_setting: The ink key setting that corresponds to zero density change, used as a reference
    - target_de: The target Delta E value for color accuracy, used in the AI prediction features.

    '''
    return render_template('index.html', config=config)
    

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    '''
    GET: Renders the settings page (settings.html) with the current configuration values for display and editing.
    POST: Updates the global config dictionary with new values submitted from the settings form, then redirects back
          to the main page. The form fields correspond to the keys in the config dictionary, allowing users to adjust 
          the density relation, uncoated factor, zero setting, and target Delta E directly from the UI.
    '''
    if request.method == 'POST':
        config["density_relation"] = float(request.form.get("density_relation", 0.02))
        config["uncoated_factor"] = float(request.form.get("uncoated_factor", 0.75))
        config["zero_setting"] = float(request.form.get("zero_setting", 0.0))
        config["target_de"] = float(request.form.get("target_de", 2.5))
        return redirect('/')
    return render_template('settings.html', config=config)

@app.route('/predict_all', methods=['POST'])
def predict_all():
    ''' 
    This endpoint processes a batch of zone data sent as JSON in the POST request. For each zone, it performs the following steps:
1. AI Prediction: It constructs a feature DataFrame for the zone based on its attributes (color, paper type, initial density, etc.) and uses the pre-loaded decision tree model to predict the optimal ink key setting.
2. Fixed Physics Relation: It calculates the predicted density using a linear relation based on the initial density and the predicted ink key setting, applying a different slope for uncoated paper types as defined in the global config.
3. Results Aggregation: It collects the predicted ink key and density for each zone, as well as calculating the average predicted density across all zones. The final results are returned as a JSON response containing the status, individual 
   zone predictions, and the average density. Error handling is included to catch and report any exceptions that occur during processing.  
    '''
    if not model: return jsonify({"status": "error", "message": "Model missing"})
    try:
        req = request.json
        results = []
        total_dens = 0
        count = 0

        for z in req.get('zones', []):
            # 1. AI Prediction
            feat_df = pd.DataFrame([{
                'Color': {'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3}.get(z['color'], 0),
                'Paper type': {'Coated': 0, 'Uncoated': 1}.get(z['paper_type'], 0),
                'Ink key zero setting': config["zero_setting"],
                'Delta E improvement': float(z['de_before']) - config["target_de"],
                'initial density': float(z['init_dens']),
                'initial ink key setting': float(z['init_key'])
            }])

            cols = ['Color', 'Paper type', 'Ink key zero setting', 'Delta E improvement', 'initial density', 'initial ink key setting']
            pred_key = float(model.predict(feat_df[cols])[0])
            
            # 2. FIXED PHYSICS RELATION
            # We apply the 'uncoated_factor' ONLY if paper is Uncoated
            current_slope = config["density_relation"]
            if z['paper_type'] == 'Uncoated':
                current_slope = current_slope * config["uncoated_factor"]

            final_dens = float(z['init_dens']) + (current_slope * (pred_key - float(z['init_key'])))
            
            total_dens += final_dens
            count += 1

            results.append({
                "zone_no": z['zone_no'], 
                "predicted_key": round(pred_key, 10),
                "predicted_density": round(final_dens, 10)
            })

        avg_density = round(total_dens / count, 10) if count > 0 else 0
        return jsonify({"status": "success", "results": results, "avg_density": avg_density})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    FlaskUI(app=app, server="flask", width=1550, height=920).run()