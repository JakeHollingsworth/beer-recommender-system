from fuzzywuzzy import process, fuzz
from flask import Flask, render_template, request, session
from utils import read_config
import pickle
from collaborative_filtering import New_User

app = Flask(__name__)
app.secret_key="lajksdhfljkafsd"
default_closest = ['Search', 'for', 'a', 'beer', 'above!']
default_recs = ['Beer A', 'Beer B', 'Beer C', 'Beer D', 'Beer E', 'Beer F', 'Beer G', 'Beer H', 'Beer I', 'Beer J']

def init_session():
    if 'closest_searches' not in session:
        session['closest_searches'] = default_closest
    if 'recommendations' not in session:
        session['recommendations'] = default_recs
    if 'selection' not in session:
        session['selection'] = None
    if 'rated_items' not in session:
        session['rated_items'] = []
    if 'item_ratings' not in session:
        session['item_ratings'] = []

def get_closest_beers(query):
    result = process.extract(query, names, limit=5,scorer=fuzz.partial_ratio)
    # Process returns (value, score) tuples
    return [match[0] for match in result]

def update_model(rated_items, item_ratings):
    mu = sum(item_ratings) / len(item_ratings)
    model_ratings = [r - mu for r in item_ratings]
    user_ml_model.add_new_rating(rated_items, model_ratings)
    return user_ml_model

@app.route("/", methods=["POST", "GET"])
def root_page():
    init_session()
    closest_searches = session['closest_searches']
    recommendations = session['recommendations']
    selection = session['selection']
    rated_items = session['rated_items']
    item_ratings = session['item_ratings']
    if request.method=="POST":
        # Search bar request
        if 'searchButton' in request.form:
            search = request.form["search"]
            closest_searches = get_closest_beers(search)
        # Click on a close search
        elif any(['closest' + str(i) in request.form for i in range(len(closest_searches))]):
            for i in range(len(closest_searches)):
                if 'closest' + str(i) in request.form:
                    break
            selection = closest_searches[i] if closest_searches[i] not in default_closest else None
        # Click on a recommendation
        elif any(['rec' + str(i) in request.form for i in range(len(recommendations))]):
            for i in range(len(recommendations)):
                if 'rec' + str(i) in request.form:
                    break
            selection = recommendations[i] if recommendations[i] not in default_recs else None
        # Clicks a rating
        elif 'rating' in request.form:
            new_rating = int(request.form['rating'][0])
            if selection:
                # Update rating if selection was already rated
                if int(name_to_id[selection]) in rated_items:
                    item_ratings[rated_items.index(int(name_to_id[selection]))] = new_rating
                # Else add a new rating
                else:
                    rated_items.append(int(name_to_id[selection]))
                    item_ratings.append(new_rating)
                update_model(rated_items, item_ratings)
                recommendations = [id_to_name[i] for i in user_ml_model.get_top_N(10)]
        # Deletes history
        elif 'restart' in request.form:
            closest_searches = default_closest
            recommendations = default_recs
            selection = None
            rated_items = []
            item_ratings = []
            user_ml_model.initialize_new_user_model()

        session['closest_searches'] = closest_searches
        session['recommendations'] = recommendations
        session['selection'] = selection
        session['rated_items'] = rated_items
        session['item_ratings'] = item_ratings
    return render_template("index.html", closest_searches=closest_searches, \
                           recommendations=recommendations, selection=selection,
                           user_ratings= item_ratings,
                           user_items=[id_to_name[item] for item in rated_items])


if __name__=='__main__':
    config = read_config()
    with open(config['dicts_file'], 'rb') as f:
        dicts = pickle.load(f)
    id_to_name = dicts['id_to_name']
    name_to_id = dicts['name_to_id']
    names = name_to_id.keys()
    user_ml_model = New_User(config['model_file'], config['learning_rate'], config['regularization_lambda'], config['epochs'])
    app.run(host=config['host_ip'],debug=True)
